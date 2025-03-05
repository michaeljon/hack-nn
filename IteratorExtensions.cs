using System;
using System.Collections.Generic;

namespace Learning.Neural.Networks
{
    internal static class Iterators
    {
        public static void Each<T>(this IEnumerable<T> enumerable, Action<T, int> action)
        {
            var index = 0;

            foreach (var item in enumerable)
            {
                action(item, index++);
            }
        }
    }
}
