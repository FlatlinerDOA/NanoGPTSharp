namespace PerceptivePyro.Whisper
{
    using System;
    using System.Collections.Generic;
    using TorchSharp.Modules;
    using F = nn.functional;

    public class MultiHeadAttention : nn.Module<(Tensor x, Tensor? xa, Tensor? mask, Dictionary<Linear, Tensor>? kv_cache), (Tensor, Tensor)>
    {
        private int n_head;
        private Linear query;
        public Linear key;
        public Linear value;
        private Linear @out;

        public MultiHeadAttention(int n_state, int n_head) : base(nameof(MultiHeadAttention))
        {
            this.n_head = n_head;
            this.query = nn.Linear(n_state, n_state);
            this.key = nn.Linear(n_state, n_state, hasBias: false);
            this.value = nn.Linear(n_state, n_state);
            this.@out = nn.Linear(n_state, n_state);
            RegisterComponents();
        }


        public override (Tensor, Tensor) forward((Tensor x, Tensor? xa, Tensor? mask, Dictionary<Linear, Tensor>? kv_cache) input)
        {
            (Tensor x, Tensor? xa, Tensor? mask, Dictionary<Linear, Tensor>? kv_cache) = input;
            var q = this.query.call(x);

            Tensor k, v;
            if (kv_cache is null || xa is null || !kv_cache.ContainsKey(this.key))
            {
                // hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
                // otherwise, perform key/value projections for this- or cross-attention as usual.
                k = this.key.call(xa is null ? x : xa);
                v = this.value.call(xa is null ? x : xa);
            }
            else
            {
                // for cross-attention, calculate keys and values once and reuse in subsequent calls.
                k = kv_cache[this.key];
                v = kv_cache[this.value];
            }

            (var wv, var qk) = this.qkv_attention(q, k, v, mask);
            return (this.@out.call(wv), qk);
        }

        /// <summary>
        /// Simplified overload that ignores masking and kv_cache.
        /// </summary>
        /// <param name="input">input tensor</param>
        /// <returns>output tensor</returns>
        public Tensor call(Tensor input) => this.call((input, null, null, null)).Item1;

        private (Tensor, Tensor) qkv_attention(Tensor q, Tensor k, Tensor v, Tensor? mask = null)
        {
            var (n_batch, n_ctx, n_state) = (q.shape[0], q.shape[1], q.shape[2]);
            var scale = Math.Pow(n_state / this.n_head, -0.25d);
            q = q.view(q.shape[0], q.shape[1], n_head, -1).permute(0, 2, 1, 3) * scale;
            k = k.view(k.shape[0], k.shape[1], n_head, -1).permute(0, 2, 3, 1) * scale;
            v = v.view(v.shape[0], v.shape[1], n_head, -1).permute(0, 2, 1, 3);

            var qk = q.dot(k);
            if (mask is not null)
            {
                qk = qk + mask[..(int)n_ctx, ..(int)n_ctx];
            }

            qk = qk.@float();

            var w = F.softmax(qk, dim: -1).to(q.dtype);
            return ((w.dot(v)).permute(0, 2, 1, 3).flatten(start_dim: 2), qk.detach());
        }
    }
}
