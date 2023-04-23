﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static System.Math;
using F = TorchSharp.torch.nn.functional;

namespace NanoGPTSharp
{
    /// <summary>
    /// One head of self attention
    /// </summary>
    internal class Head : nn.Module<Tensor, Tensor>
    {
        private Linear key;
        private Linear query;
        private Linear value;

        internal Head(int block_size, int n_embd, int head_size) : base(nameof(Head))
        {
            this.key = nn.Linear(n_embd, head_size, hasBias: false);
            this.query = nn.Linear(n_embd, head_size, hasBias: false);
            this.value = nn.Linear(n_embd, head_size, hasBias: false);
            this.register_buffer("tril", tril(ones(block_size, block_size)));  // triangular matrix of 1's for time so that the past see the future.
            this.RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            // Self attention - we are attending to our ourselves (in this case the x input).
            // Cross attention - we are attending to data from elsewhere.
            var (B, T, C) = (input.shape[0], input.shape[1], input.shape[2]); // batch, time, channels
            var k = this.key.call(input);
            var q = this.query.call(input);

            // compute attention scores (affinities)
            var wei = q.matmul(k.transpose(-2, -1)) * (Pow(C, -0.5d)); // (B, T, 16) @ (B, 16, T) ---> (B, T, T)
            wei = wei.masked_fill(this.get_buffer("tril")[..(int)T,..(int)T] == 0, float.NegativeInfinity); // mask fill wei, with the zeroes from tril replaced with -Inf. This gives wei 0's and -Inf in a triangle.
            wei = F.softmax(wei, dim: -1); // Softmax replaces -Inf with 0 and weights the zeroes evenly distributed by row.

            var v = this.value.call(input); // (B, T, C)
            var output = wei.matmul(v); // (B, T, T) @ (B, T, C) -> (B, T, C)
            return output;
        }
    }
}