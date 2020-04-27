using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Text;
using Nomad.Matrix;
using Vortex.Activation;

namespace Vortex_Tests
{
    [TestClass]
    public class VortexActivation
    {
        [TestMethod]
        public void ArctanTest()
        {
            Arctan a = new Arctan();
            
            Matrix mat = new Matrix(2,2);
            mat.InRandomize();

            Matrix act = a.Forward(mat);

            

        }
    }
}
