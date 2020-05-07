// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Nomad.Matrix;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer.Kernels
{
    public sealed class RmsPropKernel : BaseOptimizerKernel
    {
        private bool _initw;
        private Matrix _sDw;
        private bool _initb;
        private Matrix _sDb;

        public double Rho { get; set; }
        public double Epsilon { get; set; }

        public RmsPropKernel(RmsProp settings) : base(settings)
        {
            _initw = true;
            _initb = true;
            Rho = settings.Rho;
            Epsilon = settings.Epsilon;
        }

        public RmsPropKernel(double rho = 0.9, double epsilon = 0.0001, double alpha = 0.001) : base(new RmsProp(rho, epsilon, alpha))
        {
            _initw = true;
            _initb = true;
            Rho = rho;
            Epsilon = epsilon;
        }

        public override Matrix CalculateDeltaW(Matrix w, Matrix dJdW)
        {
            if (_initw)
            {
                _initw = false;
                _sDw = (Alpha * (w.Hadamard(dJdW)));
            }
            _sDw = (Rho * _sDw) + (1 - Rho) * dJdW.Hadamard(dJdW);
            Matrix mat = _sDw.Map(Math.Sqrt);
            Matrix epsilonMatrix = mat.Fill(Epsilon);
            mat.InAdd(epsilonMatrix);
            mat.InOneOver();
            mat.InHadamard(dJdW);
            return Alpha * mat;
        }

        public override Matrix CalculateDeltaB(Matrix b, Matrix dJdB)
        {
            if (_initb)
            {
                _initb = false;
                _sDb = (Alpha * (b.Hadamard(dJdB)));
            }
            _sDb = (Rho * _sDb) + (1 - Rho) * dJdB.Hadamard(dJdB);
            Matrix mat = _sDb.Map(Math.Sqrt);
            Matrix epsilonMatrix = mat.Fill(Epsilon);
            mat.InAdd(epsilonMatrix);
            mat.InOneOver();
            mat.InHadamard(dJdB);
            return Alpha * mat;
        }

        public override EOptimizerType Type() => EOptimizerType.RmsProp;
    }

    public sealed class RmsProp : Utility.Optimizer
    {
        public double Rho { get; set; }
        public double Epsilon { get; set; }
        public override EOptimizerType Type() => EOptimizerType.RmsProp;

        public RmsProp(double rho = 0.9, double epsilon = 0.0001, double alpha = 0.001) : base(alpha)
        {
            Rho = rho;
            Epsilon = epsilon;
        }
    }
}
