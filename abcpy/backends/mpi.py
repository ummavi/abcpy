import numpy as np
import cloudpickle
import multiprocessing
import time
import os
from operator import itemgetter

from mpi4py import MPI
from abcpy.backends import Backend, PDS, BDS

class IterableQueue():
    """
    An iterator over a multiprocessing.Queue().
    A normal q.get() call blocks until we get data while the get_nowait() 
    doesn't but raises on exception on empty. We break the iterator then.
    """
    def __init__(self,source_queue):
            self.source_queue = source_queue
    def __iter__(self):
        while True:
            try:
               yield self.source_queue.get_nowait()
            except Exception:
               return

    
class BackendMPIMaster(Backend):
    """Defines the behavior of the master process

    This class defines the behavior of the master process (The one
    with rank==0) in MPI.

    """

    #Define some operation codes to make it more readable
    OP_PARALLELIZE, OP_MAP, OP_COLLECT, OP_BROADCAST, OP_DELETEPDS, OP_DELETEBDS, OP_FINISH = [1, 2, 3, 4, 5, 6, 7]
    finalized = False

    try:
        os.mkdir("logs")
    except Exception as e:
        print("folder logs/ already exists")
        pass

    def __init__(self, master_node_ranks=[0]):

        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        self.master_node_ranks = master_node_ranks

        #Initialize the current_pds_id and bds_id
        self.__current_pds_id = 0
        self.__current_bds_id = 0

        #Initialize a BDS store for both master & slave.
        self.bds_store = {}


    def __command_slaves(self, command, data):
        """
        This method handles the sending of the command to the slaves
        telling them what operation to perform next.

        Parameters
        ----------
        command: operation code of OP_xxx
            One of the operation codes defined in the class definition as OP_xxx
            which tell the slaves what operation they're performing.
        data:  tuple
            Any of the data required for the operation which needs to be bundled
            in the data packet sent.
        """

        if command == self.OP_PARALLELIZE:
            #In parallelize we receive data as (pds_id)
            data_packet = (command, data[0])

        elif command == self.OP_MAP:
            #In map we receive data as (pds_id,pds_id_new,func)
            #Use cloudpickle to dump the function into a string.
            function_packed = self.__sanitize_and_pack_func(data[2])
            data_packet = (command, data[0], data[1], function_packed)

        elif command == self.OP_BROADCAST:
            data_packet = (command, data[0])

        elif command == self.OP_COLLECT:
            #In collect we receive data as (pds_id)
            data_packet = (command, data[0])

        elif command == self.OP_DELETEPDS or command == self.OP_DELETEBDS:
            #In deletepds we receive data as (pds_id) or bds_id
            data_packet = (command, data[0])

        elif command == self.OP_FINISH:
            data_packet = (command,)

        _ = self.comm.bcast(data_packet, root=0)


    def __sanitize_and_pack_func(self, func):
        """
        Prevents the function from packing the backend by temporarily
        setting it to another variable and then uses cloudpickle
        to pack it into a string to be sent.

        Parameters
        ----------
        func: Python Function
            The function we are supposed to pack while sending it along to the slaves
            during the map function

        Returns
        -------
        Returns a string of the function packed by cloudpickle

        """

        #Set the backend to None to prevent it from being packed
        globals()['backend'] = {}

        function_packed = cloudpickle.dumps(func)

        #Reset the backend to self after it's been packed
        globals()['backend'] = self

        return function_packed


    def __generate_new_pds_id(self):
        """
        This method generates a new pds_id to associate a PDS with it's remote counterpart
        that slaves use to store & index data based on the pds_id they receive

        Returns
        -------
        Returns a unique integer id.

        """

        self.__current_pds_id += 1
        return self.__current_pds_id


    def __generate_new_bds_id(self):
        """
        This method generates a new bds_id to associate a BDS with it's remote counterpart
        that slaves use to store & index data based on the bds_id they receive

        Returns
        -------
        Returns a unique integer id.

        """

        self.__current_bds_id += 1
        return self.__current_bds_id


    def parallelize(self, python_list):
        """
        This method distributes the list on the available workers and returns a
        reference object.

        The list is split into number of workers many parts as a numpy array.
        Each part is sent to a separate worker node using the MPI scatter.

        MASTER: python_list is the real data that is to be split up

        Parameters
        ----------
        list: Python list
            the list that should get distributed on the worker nodes

        Returns
        -------
        PDSMPI class (parallel data set)
            A reference object that represents the parallelized list
        """

        # print("Parallelize called on",python_list)
        # Tell the slaves to enter parallelize()
        pds_id = self.__generate_new_pds_id()
        self.__command_slaves(self.OP_PARALLELIZE, (pds_id,))

        #Initialize empty data lists for the processes on the master node
        rdd_masters = [[] for i in range(len(self.master_node_ranks))]

        #Split the data only amongst the number of workers
        rdd_slaves = np.array_split(python_list, self.size - len(self.master_node_ranks), axis=0)

        #Combine the lists into the final rdd before we split it across all ranks.
        rdd = rdd_masters + rdd_slaves
        # print("Sent data chunk",rdd)

        data_chunk = self.comm.scatter(rdd, root=0)

        pds = PDSMPI(data_chunk, pds_id, self)

        return pds


    def map(self, func, pds):
        """
        A distributed implementation of map that works on parallel data sets (PDS).

        On every element of pds the function func is called.

        Parameters
        ----------
        func: Python func
            A function that can be applied to every element of the pds
        pds: PDS class
            A parallel data set to which func should be applied

        Returns
        -------
        PDSMPI class
            a new parallel data set that contains the result of the map
        """

        # Tell the slaves to enter the map() with the current pds_id & func.
        #Get pds_id of dataset we want to operate on
        pds_id = pds.pds_id

        #Generate a new pds_id to be used by the slaves for the resultant PDS
        pds_id_new = self.__generate_new_pds_id()

        data = (pds_id, pds_id_new, func)
        self.__command_slaves(self.OP_MAP, data)

        #Master doesn't do any computations. Just have no data in the pds_res.
        pds_res = PDSMPI([], pds_id_new, self)

        return pds_res


    def collect(self, pds):
        """
        Gather the pds from all the workers,
            send it to the master and return it as a standard Python list.

        Parameters
        ----------
        pds: PDS class
            a parallel data set

        Returns
        -------
        Python list
            all elements of pds as a list
        """

        # Tell the slaves to enter collect with the pds's pds_id
        self.__command_slaves(self.OP_COLLECT, (pds.pds_id,))

        python_list = self.comm.gather(pds.python_list, root=0)


        # When we gather, the results are a list of lists one
        # .. per rank. Undo that by one level and still maintain multi
        # .. dimensional output (which is why we cannot use np.flatten)
        combined_result = []
        list(map(combined_result.extend, python_list))

        # print("Collect got",combined_result)
        return combined_result


    def broadcast(self, value):
        # Tell the slaves to enter broadcast()
        bds_id = self.__generate_new_bds_id()
        self.__command_slaves(self.OP_BROADCAST, (bds_id,))

        _ = self.comm.bcast(value, root=0)

        bds = BDSMPI(value, bds_id, self)
        return bds


    def delete_remote_pds(self, pds_id):
        """
        A public function for the PDS objects on the master to call when they go out of
        scope or are deleted in order to ensure the same happens on the slaves.

        Parameters
        ----------
        pds_id: int
            A pds_id identifying the remote PDS on the slaves to delete.
        """

        if  not self.finalized:
            self.__command_slaves(self.OP_DELETEPDS, (pds_id,))


    def delete_remote_bds(self, bds_id):
        """
        Public function for the BDS objects on the master to call when they go
        out of score or are deleted in order to ensure they are deleted
        ont he slaves as well.

        Parameters
        ----------
        bds_id: int
            A bds_id identifying the remote BDS on the slaves to delete.
        """

        if  not self.finalized:
            #The master deallocates it's BDS data. Explicit because
            #.. bds_store and BDSMPI object are disconnected.
            del backend.bds_store[bds_id]
            self.__command_slaves(self.OP_DELETEBDS, (bds_id,))


    def __del__(self):
        """
        Overriding the delete function to explicitly call MPI.finalize().
        This is also required so we can tell the slaves to get out of the
        while loop they are in and exit gracefully and they themselves call
        finalize when they die.
        """

        #Tell the slaves they can exit gracefully.
        self.__command_slaves(self.OP_FINISH, None)

        #Finalize the connection because the slaves should have finished.
        MPI.Finalize()
        self.finalized = True



def worker_target(command_q,data_q,result_q,map_in_progress,worker_id):
    '''Defines the target function the workers for every slave node enter into

    Parameters
        ----------
        command_q: multiprocessing.queue
            a queue of commands the worker listens to continuously from the node master
            commands are of the form (command_type,**data)
            (OP_DELETEBDS,bds_id) is to delete the local BDS of bds_id
            (OP_BROADCAST,bds_id,data) is to create a BDS of bds_id with data data
            (OP_MAP,func) is to perform a map function on singular items
                popped off the shared, multiprocessing data_q and place them in the
                shared result_queue
            (OP_FINISH,) is to tell the worker to break out of the loop and die.

        data_q: multiprocessing.queue
            a queue to grab data off the master while map is still going on

        result_q: multiprocessing.queue
            a queue to push the results of the worker back to the parent

        map_in_progress: multiprocessing.value
            a flag to see if a map is in progress or not. Used to decide if data_q 
            is empty because it wasn't populated fully yet or the map is really
            over. 

        worker_id: Number to correspond to a worker level id. Like how ranks were before.
    '''
    _, OP_MAP, _, OP_BROADCAST, _, OP_DELETEBDS, OP_FINISH = [1, 2, 3, 4, 5, 6, 7]

    class pseudo_worker_backend:
        """
        A pseudo backend class for the workers. 
        We set an instance of this as the global "backend" so when we access a BDS in a map and it
        tries hit globals().backend.bds_store[], it exists.
        """
        def __init__(self):
            self.bds_store = {}
            self.pds_store = {}
            self.rank = worker_id #Debugging var. Don't expect PIDs to be small like with rank ids.


    def map_over_queue():
        # print("Initiating map over queue")
        #Function to run a map over the shared queue 
        while map_in_progress.value:
            res = []
            for data_item in IterableQueue(data_q):
                item_index,item_data = data_item
                item_res = func(item_data)
                result_q.put((item_index,item_res))

    globals()['backend'] = pseudo_worker_backend()
    pid =  os.getpid()
    # print("Started worker with pid",pid," worker id:",worker_id)

    log_fd = open("logs/worker_"+str(worker_id),"w")

    while True:
        # print(pid,"Waiting for a command in queue")
        command = command_q.get()

        if command[0] == OP_MAP:
            map_start = time.time()
            # print(pid,"Got map.")
            _,func_packed = command
            func =  cloudpickle.loads(func_packed)

            #Iterate through the queue calling data_q.get()s
            #and write every result into the result_qu
            try:
                map_over_queue()
            except Exception as e:
                Exception("Worker",pid," ran into an error during map ",e)

            #Write at the end to not mess with timing results.    
            log_fd.write("MAP_START "+str(map_start)+"\nMAP_END "+str(time.time())+"\n")

        elif command[0] == OP_BROADCAST:

            #Write the BDS data directly into the bds_store
            _,bds_id,data = command
            # print(pid,"Got new BDS","with bds_id",bds_id)

            backend.bds_store[bds_id] = data

        elif command[0] == OP_DELETEBDS:
            # print(pid,"Got delete BDS")

            _,bds_id = command
            del backend.bds_store[bds_id]

        elif command[0] == OP_FINISH:
            # print(pid,"Got Finish")
            break
        else:
            print("Invalid command!")

class BackendMPISlave(Backend):
    """Defines the behavior of the slaves processes

    This class defines how the slaves should behave during operation.
    Slaves are those processes(not nodes like Spark) that have rank!=0
    and whose ids are not present in the list of non workers.
    """

    OP_PARALLELIZE, OP_MAP, OP_COLLECT, OP_BROADCAST, OP_DELETEPDS, OP_DELETEBDS, OP_FINISH = [1, 2, 3, 4, 5, 6, 7]


    def __init__(self,num_subprocesses=multiprocessing.cpu_count()):

        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        #Define the vars that will hold the pds ids received from master to operate on
        self.__rec_pds_id = None
        self.__rec_pds_id_result = None


        #Initialize a BDS store for both master & slave.
        self.bds_store = {}

        #Define a list to hold the private queues for each worker proc.
        self.worker_processes = []
        self.worker_command_queues = []

        #Define the shared queues/vars the worker procs need.
        self.data_q = multiprocessing.Queue()
        self.map_in_progress = multiprocessing.Value('b',0)
        self.result_q = multiprocessing.Queue()

        # self.log_fd = open("logs/node_"+str(self.rank),"w")

        for i in range(num_subprocesses):
            #Initialize a local(private) command queue for each subproc
            command_q = multiprocessing.Queue()

            worker_id = self.rank*100+i
            #Create a process, tell it to start running the worker_target function
            # and pass it all it's shared queues/datas
            p = multiprocessing.Process(target=worker_target, args=(command_q,self.data_q,self.result_q,self.map_in_progress,worker_id))
            p.start()

            #Save the private queues for each subproc so we can communicate
            self.worker_processes+=[p]
            self.worker_command_queues+=[command_q]

        #Go into an infinite loop waiting for commands from the user.
        self.slave_run()



    def __broadcast_command_to_workers(self,command):
        """
        This method iterates through all the workers and places the command
        command in their command_queus 
        """
        # print("Broadcasting command",command)
        for i,q in enumerate(self.worker_command_queues):
            # print ("Pushing command to process",i)
            q.put(command)


    def slave_run(self):
        """
        This method is the infinite loop a slave enters directly from init.
        It makes the slave wait for a command to perform from the master and
        then calls the appropriate function.

        This method also takes care of the synchronization of data between the
        master and the slaves by matching PDSs based on the pds_ids sent by the master
        with the command.

        Commands received from the master are of the form of a tuple.
        The first component of the tuple is always the operation to be performed
        and the rest are conditional on the operation.

        (op,pds_id) where op == OP_PARALLELIZE for parallelize
        (op,pds_id, pds_id_result, func) where op == OP_MAP for map.
        (op,pds_id) where op == OP_COLLECT for a collect operation
        (op,pds_id) where op == OP_DELETEPDS for a delete of the remote PDS on slaves
        (op,) where op==OP_FINISH for the slave to break out of the loop and terminate
        """

        # Initialize PDS data store here because only slaves need to do it.
        self.pds_store = {}

        while True:
            data = self.comm.bcast(None, root=0)

            op = data[0]
            if op == self.OP_PARALLELIZE:
                pds_id = data[1]
                self.__rec_pds_id = pds_id
                pds = self.parallelize([])
                self.pds_store[pds.pds_id] = pds


            elif op == self.OP_MAP:
                pds_id, pds_id_result, function_packed = data[1:]
                self.__rec_pds_id, self.__rec_pds_id_result = pds_id, pds_id_result

                #Use cloudpickle to convert back function string to a function
                # func = cloudpickle.loads(function_packed)
                #Set the function's backend to current class
                #so it can access bds_store properly
                # func.backend = self


                # Access an existing PDS
                pds = self.pds_store[pds_id]
                pds_res = self.map(function_packed, pds)

                # Store the result in a newly gnerated PDS pds_id
                self.pds_store[pds_res.pds_id] = pds_res

            elif op == self.OP_BROADCAST:
                self.__bds_id = data[1]
                self.broadcast(None)

            elif op == self.OP_COLLECT:
                pds_id = data[1]

                # Access an existing PDS from data store
                pds = self.pds_store[pds_id]

                self.collect(pds)

            elif op == self.OP_DELETEPDS:
                pds_id = data[1]
                del self.pds_store[pds_id]

            elif op == self.OP_DELETEBDS:
                bds_id = data[1]
                del self.bds_store[bds_id]

                #Tell the worker processes to delete a BDS
                data_packet = (self.OP_DELETEBDS,bds_id)
                self.__broadcast_command_to_workers(data_packet)

            elif op == self.OP_FINISH:
                #Tell the worker processes to break out of their loops and die.
                data_packet = (self.OP_FINISH,)
                self.__broadcast_command_to_workers(data_packet)

                #Wait for the processes to finish
                for p in self.worker_processes:
                    p.join()

                #Quit as usual
                quit()
            else:
                raise Exception("Slave recieved unknown command code")


    def __get_received_pds_id(self):
        """
        Function to retrieve the pds_id(s) we received from the master to associate
        our slave's created PDS with the master's.
        """

        return self.__rec_pds_id, self.__rec_pds_id_result


    def parallelize(self, python_list):
        """
        This method distributes the list on the available workers and returns a
        reference object.

        The list is split into number of workers many parts as a numpy array.
        Each part is sent to a separate worker node using the MPI scatter.

        SLAVE: python_list should be [] and is ignored by the scatter()

        Parameters
        ----------
        list: Python list
            the list that should get distributed on the worker nodes

        Returns
        -------
        PDSMPI class (parallel data set)
            A reference object that represents the parallelized list
        """

        #Get the PDS id we should store this data in
        pds_id, pds_id_new = self.__get_received_pds_id()

        data_chunk = self.comm.scatter(None, root=0)

        # print("Slave got data chunk",data_chunk)
        pds = PDSMPI(data_chunk, pds_id, self)

        return pds


    def map(self, func, pds):
        """
        A distributed implementation of map that works on parallel data sets (PDS).

        On every element of pds the function func is called.

        Parameters
        ----------
        func: Python func
            A function that can be applied to every element of the pds
        pds: PDS class
            A parallel data set to which func should be applied

        Returns
        -------
        PDSMPI class
            a new parallel data set that contains the result of the map
        """

        MAP_START = time.time()
        #Get the PDS id we operate on and the new one to store the result in
        pds_id, pds_id_new = self.__get_received_pds_id()


        #Initialize a common data_q and result_q for all the slaves to pop data off
        #  and to place their results in
        total_data_elements = len(pds.python_list)

        #Create a data packet to send the workers
        worker_data_packet = (self.OP_MAP,func)

        #So the workers know a map is in progress
        self.map_in_progress.value = True


        #Send it off to the workers
        self.__broadcast_command_to_workers(worker_data_packet)

        MAP_SENT_COMMAND = time.time()

        # print("Finished brodcasting. Now populating with ",total_data_elements)
        #Populate the data_q with the data that needs to be distributed
        for item_index,item_data in enumerate(pds.python_list):
            self.data_q.put((item_index,item_data))

        MAP_SENT_DATA = time.time()


        # print ("Finished populating. Waiting for result_q to fill up")
        #Wait till the result queue is fully populated 
        rdd = []
        rdd_indices = []

        while len(rdd)<total_data_elements:
            # print("len(rdd)",len(rdd),"/",total_data_elements)
            for e in  IterableQueue(self.result_q):
                item_index,item_res = e
                # print("Got a result",e)
                rdd_indices+=[item_index]
                rdd+=[item_res]

        MAP_REVC_DATA = time.time()

        self.map_in_progress.value = False

        # print("Got rdd",rdd)
        # print("got rdd indices",rdd_indices)

        #Sort the RDD back to the right order
        rdd_sorted = [rdd[i] for i in np.argsort(rdd_indices)]

        # print("Rdd sorted",rdd_sorted)
        pds_res = PDSMPI(rdd_sorted, pds_id_new, self)
        MAP_DONE = time.time()


        # data =    "MAP_START "+str(MAP_START)+"\nMAP_SENT_COMMAND "+str(MAP_SENT_COMMAND)+"\nMAP_SENT_DATA "+str(MAP_SENT_DATA)+"\nMAP_REVC_DATA "+str(MAP_REVC_DATA)+"\nMAP_DONE "+str(MAP_DONE)+"\n"
        # self.log_fd.write(data)
        return pds_res


    def collect(self, pds):
        """
        Gather the pds from all the workers,
        send it to the master and return it as a standard Python list.

        Parameters
        ----------
        pds: PDS class
            a parallel data set

        Returns
        -------
        Python list
            all elements of pds as a list
        """

        #Send the data we have back to the master
        _ = self.comm.gather(pds.python_list, root=0)


    def broadcast(self, value):
        """
        Value is ignored for the slaves. We get data from master
        """
        value = self.comm.bcast(None, root=0)
        self.bds_store[self.__bds_id] = value

        #Tell the workers a new BDS has arrived
        data_packet = (self.OP_BROADCAST,self.__bds_id,value)
        self.__broadcast_command_to_workers(data_packet)


class BackendMPI(BackendMPIMaster if MPI.COMM_WORLD.Get_rank() == 0 else BackendMPISlave):
    """A backend parallelized by using MPI

    The backend conditionally inherits either the BackendMPIMaster class
    or the BackendMPISlave class depending on it's rank. This lets
    BackendMPI have a uniform interface for the user but allows for a
    logical split between functions performed by the master
    and the slaves.
    """

    def __init__(self, master_node_ranks=[0]):
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        if self.size < 2:
            raise ValueError('A minimum of 2 ranks are required for the MPI backend')


        #Set the global backend
        globals()['backend'] = self


        #Call the appropriate constructors and pass the required data
        if self.rank == 0:
            super().__init__(master_node_ranks)
        else:
            super().__init__()
            raise Exception("Slaves exitted main loop.")


class PDSMPI(PDS):
    """
    This is an MPI wrapper for a Python parallel data set.
    """

    def __init__(self, python_list, pds_id, backend_obj):
        self.python_list = python_list
        self.pds_id = pds_id
        self.backend_obj = backend_obj

    def __del__(self):
        """
        Destructor to be called when a PDS falls out of scope and/or is being deleted.
        Uses the backend to send a message to destroy the slaves' copy of the pds.
        """
        try:
            self.backend_obj.delete_remote_pds(self.pds_id)
        except AttributeError:
            #Catch "delete_remote_pds not defined" for slaves and ignore.
            pass


class BDSMPI(BDS):
    """
    This is a wrapper for MPI's BDS class.
    """

    def __init__(self, object, bds_id, backend_obj):
        #The BDS data is no longer saved in the BDS object.
        #It will access & store the data only from the current backend
        # print("BDS created called for rank",backend.rank,os.getpid(),"ID:",bds_id)

        self.bds_id = bds_id
        backend.bds_store[self.bds_id] = object
        # self.backend_obj = backend_obj

    def value(self):
        """
        This method returns the actual object that the broadcast data set represents.
        """
        # print("BDS Value called for rank",backend.rank,os.getpid())
        return backend.bds_store[self.bds_id]

    def __del__(self):
        """
        Destructor to be called when a BDS falls out of scope and/or is being deleted.
        Uses the backend to send a message to destroy the slaves' copy of the bds.
        """

        try:
            backend.delete_remote_bds(self.bds_id)
        except AttributeError:
            #Catch "delete_remote_pds not defined" for slaves and ignore.
            pass

class BackendMPITestHelper:
    """
    Helper function for some of the test cases to be able to access and verify class members.
    """
    def check_pds(self, k):
        """Checks if a PDS exists in the pds data store. Used to verify deletion and creation
        """
        return k in backend.pds_store.keys()

    def check_bds(self, k):
        """Checks if a BDS exists in the BDS data store. Used to verify deletion and creation
        """
        return k in backend.bds_store.keys()
