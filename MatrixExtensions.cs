using System;
using System.Numerics;
using MathNet.Numerics.LinearAlgebra;

namespace Learning.Neural.Networks
{
    internal static class MatrixExtensions
    {
        public static string ShapeAsString<T>(this Matrix<T> matrix) where T : struct, IFormattable, IEquatable<T>
        {
            return $"({matrix.RowCount}, {matrix.ColumnCount})";
        }

        public static (int rowCount, int columnCount) Shape<T>(this Matrix<T> matrix) where T : struct, IFormattable, IEquatable<T>
        {
            return (matrix.RowCount, matrix.ColumnCount);
        }

        public static void AssertShape<T>(this Matrix<T> matrix, int rowCount, int columnCount) where T : struct, IFormattable, IEquatable<T>
        {
            if (matrix.RowCount == rowCount && matrix.ColumnCount == columnCount)
            {
                return;
            }

            throw new Exception($"matrix size {matrix.ShapeAsString()} is not ({rowCount}, {columnCount})");
        }

        public static Matrix<T> Broadcast<T>(this Matrix<T> matrix, int columnCount) where T : struct, IFormattable, IEquatable<T>
        {
            return Matrix<T>.Build.Dense(matrix.RowCount, columnCount, (i, j) => matrix[i, 0]);
        }

        public static Matrix<T> SumAcrossRows<T>(this Matrix<T> matrix) where T : struct, IFormattable, IEquatable<T>, INumber<T>
        {
            T[] values = new T[matrix.RowCount];

            for (var row = 0; row < matrix.RowCount; row++)
            {
                T sum = default;

                for (var col = 0; col < matrix.ColumnCount; col++)
                {
                    sum += matrix[row, col];
                }

                values[row] = sum;
            }

            return MathNet.Numerics.LinearAlgebra.Vector<T>.Build.DenseOfArray(values).ToColumnMatrix();
        }

        public static void Print<T>(this Matrix<T> matrix) where T : struct, IFormattable, IEquatable<T>
        {
            Console.Write("[");
            for (var r = 0; r < matrix.RowCount; r++)
            {
                Console.Write(r == 0 ? "[ " : " [ ");

                for (var c = 0; c < matrix.ColumnCount; c++)
                {
                    Console.Write($"{matrix[r, c]:F8} ");
                }

                if (r == matrix.RowCount - 1)
                {
                    Console.WriteLine("]]");
                }
                else
                {
                    Console.WriteLine("]");
                }
            }
        }
    }
}
