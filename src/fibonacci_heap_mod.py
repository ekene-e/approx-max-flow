'''
(Borrowed from Data Structures course.)

An implementation of a priority queue backed by a Fibonacci heap, as described
by Fredman and Tarjan.  Fibonacci heaps are interesting theoretically because
they have asymptotically good runtime guarantees for many operations.  In
particular, insert, peek, and decrease-key all run in amortized O(1) time.
dequeue_min and delete each run in amortized O(lg n) time.  This allows
algorithms that rely heavily on decrease-key to gain significant performance
boosts.  For example, Dijkstra's algorithm for single-source shortest paths can
be shown to run in O(m + n lg n) using a Fibonacci heap, compared to O(m lg n)
using a standard binary or binomial heap.

Internally, a Fibonacci heap is represented as a circular, doubly-linked list
of trees obeying the min-heap property.  Each node stores pointers to its
parent (if any) and some arbitrary child.  Additionally, every node stores its
degree (the number of children it has) and whether it is a "marked" node.
Finally, each Fibonacci heap stores a pointer to the tree with the minimum
value.

To insert a node into a Fibonacci heap, a singleton tree is created and merged
into the rest of the trees.  The merge operation works by simply splicing
together the doubly-linked lists of the two trees, then updating the min
pointer to be the smaller of the minima of the two heaps.  Peeking at the
smallest element can therefore be accomplished by just looking at the min
element.  All of these operations complete in O(1) time.

The tricky operations are dequeue_min and decrease_key.  dequeue_min works by
removing the root of the tree containing the smallest element, then merging its
children with the topmost roots.  Then, the roots are scanned and merged so
that there is only one tree of each degree in the root list.  This works by
maintaining a dynamic array of trees, each initially null, pointing to the
roots of trees of each dimension.  The list is then scanned and this array is
populated.  Whenever a conflict is discovered, the appropriate trees are merged
together until no more conflicts exist.  The resulting trees are then put into
the root list.  A clever analysis using the potential method can be used to
show that the amortized cost of this operation is O(lg n), see "Introduction to
Algorithms, Second Edition" by Cormen, Rivest, Leiserson, and Stein for more
details.

The other hard operation is decrease_key, which works as follows.  First, we
update the key of the node to be the new value.  If this leaves the node
smaller than its parent, we're done.  Otherwise, we cut the node from its
parent, add it as a root, and then mark its parent.  If the parent was already
marked, we cut that node as well, recursively mark its parent, and continue
this process.  This can be shown to run in O(1) amortized time using yet
another clever potential function.  Finally, given this function, we can
implement delete by decreasing a key to -infinity, then calling dequeue_min to
extract it.
'''

import math
import collections


def merge_lists(one, two):
    '''
    Utility function which, given two pointers into disjoint circularly-
    linked lists, merges the two lists together into one circularly-linked
    list in O(1) time.  Because the lists may be empty, the return value
    is the only pointer that's guaranteed to be to an element of the
    resulting list.

    This function assumes that one and two are the minimum elements of the
    lists they are in, and returns a pointer to whichever is smaller.  If
    this condition does not hold, the return value is some arbitrary pointer
    into the doubly-linked list.

    @param one A pointer into one of the two linked lists.
    @param two A pointer into the other of the two linked lists.
    @return A pointer to the smallest element of the resulting list.
    '''
    if one is None and two is None:
        return None
    elif one is not None and two is None:
        return one
    elif one is None and two is not None:
        return two
    else:
        # Both non-None; actually do the splice.

        # This is actually not as easy as it seems.  The idea is that we'll
        # have two lists that look like this:
        #
        # +----+     +----+     +----+
        # |    |--N->|one |--N->|    |
        # |    |<-P--|    |<-P--|    |
        # +----+     +----+     +----+
        #
        #
        # +----+     +----+     +----+
        # |    |--N->|two |--N->|    |
        # |    |<-P--|    |<-P--|    |
        # +----+     +----+     +----+
        #
        # And we want to relink everything to get
        #
        # +----+     +----+     +----+---+
        # |    |--N->|one |     |    |   |
        # |    |<-P--|    |     |    |<+ |
        # +----+     +----+<-\  +----+ | |
        #                  \  P        | |
        #                   N  \       N |
        # +----+     +----+  \->+----+ | |
        # |    |--N->|two |     |    | | |
        # |    |<-P--|    |     |    | | P
        # +----+     +----+     +----+ | |
        #              ^ |             | |
        #              | +-------------+ |
        #              +-----------------+

        one_next = one.m_next

        one.m_next = two.m_next
        one.m_next.m_prev = one
        two.m_next = one_next
        two.m_next.m_prev = two

        if one.m_priority < two.m_priority:
            return one
        else:
            return two


def merge(one, two):
    '''
    Given two Fibonacci heaps, returns a new Fibonacci heap that contains
    all of the elements of the two heaps.  Each of the input heaps is
    destructively modified by having all its elements removed.  You can
    continue to use those heaps, but be aware that they will be empty
    after this call completes.

    @param one The first Fibonacci heap to merge.
    @param two The second Fibonacci heap to merge.
    @return A new Fibonacci_heap containing all of the elements of both
            heaps.
    '''
    result = Fibonacci_heap()

    result.m_min = merge_lists(one.m_min, two.m_min)

    result.m_size = one.m_size + two.m_size

    # Clear the old heaps.
    one.m_size = two.m_size = 0
    one.m_min = None
    two.m_min = None

    return result

class Entry(object):
    '''Hold an entry in the heap'''

    __slots__ = ['m_degree', 'm_is_marked', 'm_parent', 'm_child', 'm_next', 'm_prev', 'm_elem', 'm_priority']

    def __init__(self, elem, priority):
        self.m_degree = 0
        self.m_is_marked = False

        self.m_parent = None
        self.m_child = None

        self.m_next = self.m_prev = self
        self.m_elem = elem
        self.m_priority = priority

    def __lt__(self, other):
        if self.m_priority < other.m_priority:
            return True
        else:
            if self.m_elem < other.m_elem:
                return True
            else:
                return False

    def __eq__(self, other):
        if self.m_priority == other.m_priority:
            return True
        else:
            if self.m_elem == other.m_elem:
                return True
            else:
                return False

    def __gt__(self, other):
        if self.m_priority > other.m_priority:
            return True
        else:
            if self.m_elem > other.m_elem:
                return True
            else:
                return False

    def __cmp__(self, other):
        if self.__lt__(other):
            return -1
        elif self.__gt__(other):
            return 1
        else:
            return 0

    def get_value(self):
        '''
        Returns the element represented by this heap entry.

        @return The element represented by this heap entry.
        '''
        return self.m_elem

    def set_value(self, value):
        '''
        Sets the element associated with this heap entry.

        @param value The element to associate with this heap entry.
        '''
        self.m_elem = value

    def get_priority(self):
        '''
        Returns the priority of this element.

        @return The priority of this element.
        '''
        return self.m_priority

    def _entry(self, elem, priority):
        '''
        Constructs a new Entry that holds the given element with the indicated
        priority.

        @param elem The element stored in this node.
        @param priority The priority of this element.
        '''
        self.m_next = self.m_prev = self
        self.m_elem = elem
        self.m_priority = priority


class Fibonacci_heap(object):
    '''
    A class representing a Fibonacci heap.

    @param T The type of elements to store in the heap.
    @author Keith Schwarz (htiek@cs.stanford.edu)
    '''
    def __init__(self):
        self.m_min = None
        self.m_size = 0

    def enqueue(self, value, priority):
        '''
        Inserts the specified element into the Fibonacci heap with the specified
        priority.  Its priority must be a valid double, so you cannot set the
        priority to NaN.

        @param value The value to insert.
        @param priority Its priority, which must be valid.
        @return An Entry representing that element in the tree.
        '''
        self._check_priority(priority)
        result = Entry(value, priority)
        self.m_min = merge_lists(self.m_min, result)
        self.m_size += 1
        return result

    def min(self):
        '''
        Returns an Entry object corresponding to the minimum element of the
        Fibonacci heap, raising an IndexError if the heap is
        empty.

        @return The smallest element of the heap.
        @throws IndexError If the heap is empty.
        '''
        if not bool(self):
            raise IndexError("Heap is empty.")
        return self.m_min

    def __bool__(self):
        '''
        Returns whether the heap is nonempty.

        @return Whether the heap is nonempty.
        '''
        if self.m_min is None:
            return False
        else:
            return True

    __nonzero__ = __bool__

    def __len__(self):
        '''
        Returns the number of elements in the heap.

        @return The number of elements in the heap.
        '''
        return self.m_size

    def dequeue_min(self):
        '''
        Dequeues and returns the minimum element of the Fibonacci heap.  If the
        heap is empty, this throws an IndexError.

        @return The smallest element of the Fibonacci heap.
        @throws IndexError if the heap is empty.
        '''
        if not bool(self):
            raise IndexError("Heap is empty.")
        self.m_size -= 1
        min_elem = self.m_min
        if self.m_min.m_next is self.m_min:
            self.m_min = None
        else:
            self.m_min.m_prev.m_next = self.m_min.m_next
            self.m_min.m_next.m_prev = self.m_min.m_prev
            self.m_min = self.m_min.m_next
        if min_elem.m_child is not None:
            curr = min_elem.m_child
            while True:
                curr.m_parent = None
                curr = curr.m_next
                if curr is min_elem.m_child:
                    break
        self.m_min = merge_lists(self.m_min, min_elem.m_child)
        if self.m_min is None:
            return min_elem
        tree_table = collections.deque()
        to_visit = collections.deque()
        curr = self.m_min
        while not to_visit or to_visit[0] is not curr:
            to_visit.append(curr)
            curr = curr.m_next
        for curr in to_visit:
            while True:
                while curr.m_degree >= len(tree_table):
                    tree_table.append(None)
                if tree_table[curr.m_degree] is None:
                    tree_table[curr.m_degree] = curr
                    break
                other = tree_table[curr.m_degree]
                tree_table[curr.m_degree] = None
                if other.m_priority < curr.m_priority:
                    minimum = other
                else:
                    minimum = curr
                if other.m_priority < curr.m_priority:
                    maximum = curr
                else:
                    maximum = other
                maximum.m_next.m_prev = maximum.m_prev
                maximum.m_prev.m_next = maximum.m_next
                maximum.m_next = maximum.m_prev = maximum
                minimum.m_child = merge_lists(minimum.m_child, maximum)
                maximum.m_parent = minimum
                maximum.m_is_marked = False
                minimum.m_degree += 1
                curr = minimum
            if curr.m_priority <= self.m_min.m_priority:
                self.m_min = curr
        return min_elem

    def decrease_key(self, entry, new_priority):
        '''
        Decreases the key of the specified element to the new priority.  If the
        new priority is greater than the old priority, this function raises an
        ValueError.  The new priority must be a finite double,
        so you cannot set the priority to be NaN, or +/- infinity.  Doing
        so also raises an ValueError.

        It is assumed that the entry belongs in this heap.  For efficiency
        reasons, this is not checked at runtime.

        @param entry The element whose priority should be decreased.
        @param newPriority The new priority to associate with this entry.
        @throws ValueError If the new priority exceeds the old
                priority, or if the argument is not a finite double.
        '''
        self._check_priority(new_priority)
        if new_priority > entry.m_priority:
            raise ValueError("New priority exceeds old.")
        self.decrease_key_unchecked(entry, new_priority)

    def delete(self, entry):
        '''
        Deletes this Entry from the Fibonacci heap that contains it.

        It is assumed that the entry belongs in this heap.  For efficiency
        reasons, this is not checked at runtime.

        @param entry The entry to delete.
        '''
        self.decrease_key_unchecked(entry, float("-inf"))
        self.dequeue_min()

    @staticmethod
    def _check_priority(priority):
        '''
        Utility function which, given a user-specified priority, checks whether
        it's a valid double and throws an ValueError otherwise.

        @param priority The user's specified priority.
        @throws ValueError if it is not valid.
        '''
        if math.isnan(priority) or math.isinf(priority):
            raise ValueError("Priority {} is invalid.".format(priority))

    def decrease_key_unchecked(self, entry, priority):
        '''
        Decreases the key of a node in the tree without doing any checking to ensure
        that the new priority is valid.

        @param entry The node whose key should be decreased.
        @param priority The node's new priority.
        '''
        entry.m_priority = priority
        if entry.m_parent is not None and entry.m_priority <= entry.m_parent.m_priority:
            self.cut_node(entry)
        if entry.m_priority <= self.m_min.m_priority:
            self.m_min = entry

    def cut_node(self, entry):
        '''
        Cuts a node from its parent.  If the parent was already marked, recursively
        cuts that node from its parent as well.

        @param entry The node to cut from its parent.
        '''
        entry.m_is_marked = False

        if entry.m_parent is None:
            return
        if entry.m_next is not entry:
            entry.m_next.m_prev = entry.m_prev
            entry.m_prev.m_next = entry.m_next
        if entry.m_parent.m_child is entry:
            if entry.m_next is not entry:
                entry.m_parent.m_child = entry.m_next
            else:
                entry.m_parent.m_child = None
        entry.m_parent.m_degree -= 1
        entry.m_prev = entry.m_next = entry
        self.m_min = merge_lists(self.m_min, entry)
        if entry.m_parent.m_is_marked:
            self.cut_node(entry.m_parent)
        else:
            entry.m_parent.m_is_marked = True
        entry.m_parent = None
