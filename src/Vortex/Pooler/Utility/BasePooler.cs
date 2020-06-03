using System;
using System.Collections.Generic;
using System.Text;
using Nomad.Core;

namespace Vortex.Pooler.Utility
{
    public abstract class BasePooler : IPooler
    {
        public int StrideRow { get; set; }
        public int StrideCol { get; set; }
        public int PadSizeRow { get; set; }
        public int PadSizeCol { get; set; }
        public double PadValue { get; set; }

        protected BasePooler(int strideRow = 1, int strideCol = 1, int padSizeRow = 1, int padSizeCol = 1, double padValue = 0.0)
        {
            StrideRow = strideRow;
            StrideCol = strideCol;
            PadSizeRow = padSizeRow;
            PadSizeCol = padSizeCol;
            PadValue = padValue;
        }

        public Matrix Pad(Matrix x, Matrix filter)
        {
            var sizeRow = (int)((filter.Rows - 1.0) / 2.0);
            var sizeCol = (int)((filter.Columns - 1.0) / 2.0);

            var mat = new Matrix(x.Rows + sizeRow * 2, x.Columns + sizeCol * 2);
            mat.InFill(PadValue);

            for (var i = sizeRow; i < mat.Rows - sizeRow; i++)
            for (var j = sizeCol; j < mat.Columns - sizeCol; j++) 
                mat[i, j] = x[i - sizeRow, j - sizeCol];

            return mat;
        }

        public abstract Matrix Pool(Matrix x, Matrix filter);

        public abstract EPoolerType Type();
    }
}
