#ifndef __CXX_SORT_ALGORITHM_H
#define __CXX_SORT_ALGORITHM_H

#include <iterator>
#ifndef _CXX_SYMBOL
#include "cu-symbol.h"
#endif

template <typename T>
_CXX_EXPORT inline void __SORT_line(std::iterator_traits<T> __T_begin,
                                    const std::iterator_traits<T> __T_end);

template <typename T>
_CXX_EXPORT inline void __SORT_word(std::iterator_traits<T> __T_begin,
                                    std::iterator_traits<T> __T_end);

template <typename T>
_CXX_EXPORT inline void __SORT_char(std::iterator_traits<T> __T_begin,
                                    std::iterator_traits<T> __T_end);

#endif // __CXX_SORT_ALGORITHM_H
