// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using System.Collections.Generic;
using Nomad.Core;
using Vortex.Pooler.Utility;

namespace Vortex.Pooler.Kernels
{
    public class Sum : BasePooler
    {
        public Sum(int strideRow = 1, int strideCol = 1, int padSizeRow = 1, int padSizeCol = 1, double padValue = 0.0) 
            : base(strideRow, strideCol, padSizeRow, padSizeCol,padValue)
        {
        }

        public override Matrix Pool(Matrix x, Matrix filter)
        {

            var newX = (x.Rows - filter.Rows) / StrideRow + 1;
            var newY = (x.Columns - filter.Columns) / StrideCol + 1;
            var result = new Matrix(newX, newY);

            var resArray = new List<double>();
            for (var i = 0; i < x.Rows; i += StrideRow)
            for (var j = 0; j < x.Columns; j += StrideCol)
            {
                var val = 0.0;
                for (var k = 0; k < filter.Rows; k++)
                {
                    if (k + i >= x.Rows) continue;
                    for (var l = 0; l < filter.Columns; l++)
                        if (l + j < x.Columns)
                            val += x[i + k, j + l] * filter[k, l];
                }

                resArray.Add(val);
            }

            var n = 0;
            for (var i = 0; i < result.Rows; i++)
            for (var j = 0; j < result.Columns; j++)
                result[i, j] = resArray[n++];

            return result;
        }

        public override EPoolerType Type()
        {
            return EPoolerType.Sum;
        }
    }
}
