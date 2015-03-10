'''
Created on 18-Feb-2015

@author: adarsh
Sparse vector implementation for word and NLP related fun stuff.

'''
from collections import Mapping
import _heapq
from operator import itemgetter as _itemgetter


class counting_sparse_vec(dict):
    
    def __init__(self, text="", symbol_filter="", stop_words=None):
        """
        Pass in a list of text, and this converts the text
        into a sparse representation like described below:
        If my text is the following, 
        
        "I'm just a poor boy, nobody loves me.\n
        He's just a poor boy from a poor family,\n
        Spare him his life from this monstrosity."
        
        The output will be a dict like so:
        
        {'a': 3, 'poor': 3, 'from': 2, 'just':
         2, 'life': 1, 'me.': 1, He s: 1, 'this': 1, 
        'family,': 1, 'boy': 
        1, 'nobody': 1, 'boy,': 1,
        'his': 1, 'Spare': 1, 'loves': 1,
       "I'm": 1, 'monstrosity.': 1, 'him': 1}
        """
        self.norm_squared = 0
        self.update(text)


    def __missing__(self, key):
        """ if a key is missing return 0 by default"""
        return 0
    @classmethod
    def fromkeys(cls, iterable, v=None):
        # There is no equivalent method for counters because setting v=1
        # means that no element can have a count greater than one.
        raise NotImplementedError(
            'Counter.fromkeys() is undefined.  Use counting_sparse_vec(iterable) instead.')

    def update(self, iterable_):
        """Keeps count of any item. This should be called only once while initializing."""
        if iterable_ is not None:
            if isinstance(iterable_, Mapping):
                if self:
                    for word, count in iterable_.iteritems():
                        self.norm_squared -= (self[word] ** 2)
                        self[word] = count + self.get(word, 0)
                        self.norm_squared += (self[word] ** 2)
                else:
                    """strict about this"""
                    return NotImplemented
            else:
                for i in iterable_:
                    self[i] += float(1)
                    self.norm_squared += 1

    def most_common(self, n):
        """return items with highest values"""
        if n is None:
            return sorted(self.items(), key=_itemgetter(1), reverse=True)
        else:
            _heapq.nlargest(n, self.items(), key=_itemgetter(1))

    def __repr__(self):
        """String repr. same as that in the Counter class found at
        https://hg.python.org/cpython/file/3.4/Lib/collections/__init__.py 
        """
        if not self:
            return '%s()' % self.__class__.__name__
        try:
            items = ', '.join(map('%r: %r'.__mod__, self.most_common()))
            return '%s({%s})' % (self.__class__.__name__, items)
        except TypeError:
            # handle case where values are not orderable
            return '{0}({1!r})'.format(self.__class__.__name__, dict(self))

    def __add__(self, other):
        """Strip zero counts and add one vector to another"""
        if not isinstance(other, counting_sparse_vec):
            return NotImplemented
        result = counting_sparse_vec()
        for elem, count in self.items():
            newcount = float(count + other[elem])
            if newcount != 0:
                result[elem] = newcount

        for elem, count in other.items():
            if elem not in self and count != 0:
                result[elem] = float(count)
        
        return result
    
    # __radd__=__add__
    
    def __sub__(self, other):
        """Strip zero counts and subtract one vector to another"""
        if not isinstance(other, counting_sparse_vec):
            return NotImplemented
        result = counting_sparse_vec()
        for elem, count in self.items():
            newcount = float(count - other[elem])
            if newcount != 0:
                result[elem] = newcount

        for elem, count in other.items():
            if elem not in self and count != 0:
                result[elem] = float(0.0 - count)
   
        return result

    # __rsub__=__sub__

    def __mul__(self, other):
        """element wise multiplication and scalar multiplication"""
        if isinstance(other, counting_sparse_vec):
            result = counting_sparse_vec()
            for elem, count in self.items():
                newcount = float(count * other[elem])
                if newcount != 0:
                    result[elem] = newcount

        elif isinstance(other, int) or isinstance(other, float):
            result = counting_sparse_vec()
            for elem, count in self.items():
                newcount = float(count * other)
                if newcount != 0:
                    result[elem] = newcount
        else:
            return NotImplemented

        return result
    
    def __rmul__(self, other):
        """element wise multiplication and scalar multiplication"""
        if isinstance(other, counting_sparse_vec):
            result = counting_sparse_vec()
            for elem, count in other.items():
                newcount = float(count * self[elem])
                if newcount != 0:
                    result[elem] = newcount

        elif isinstance(other, int) or isinstance(other, float):
            result = counting_sparse_vec()
            for elem, count in self.items():
                newcount = float(count * other)
                if newcount != 0:
                    result[elem] = newcount
        else:
            return NotImplemented

        return result
 
    def dot(self, other):
        if isinstance(other, counting_sparse_vec):
            if(len(other) < len(self)):
                return sum(self[elem] * other[elem] for elem in other)
            else:
                return sum(self[elem] * other[elem] for elem in self)
        else:
            return NotImplemented
    
    def scale_and_increment(self, scale, other):
        '''
        Implements self = self + scale * other
        use this over self+=scale*other
        '''
        for elem, count in other.items():
            self[elem] = self[elem] + count * scale

    def __neg__(self):
        """negate the vector"""
        return counting_sparse_vec() - self
    
    def __pos__(self):
        return counting_sparse_vec() + self
     
    def _keep_non_zero(self):
        '''Internal method to strip elements with a negative or zero count'''
        zeros = [elem for elem, count in self.items() if count == 0]
        for elem in zeros:
            del self[elem]
        return self
    
    def __iadd__(self, other):
        """
        Inplace add from another couting_sparse_vector, keeping only positive counts.
        """
        for elem, count in other.items():
            self.norm_squared -= (self[elem] ** 2)
            self[elem] += count
            self.norm_squared += (self[elem] ** 2)
        return self._keep_non_zero()
    
    def __isub__(self, other):
        """
        Inplace add from another couting_sparse_vector, keeping only positive counts.
        """
        for elem, count in other.items():
            self.norm_squared -= (self[elem] ** 2)
            self[elem] -= count
            self.norm_squared += (self[elem] ** 2)            
        return self._keep_non_zero()
