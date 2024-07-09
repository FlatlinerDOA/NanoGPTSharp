namespace PerceptivePyro.Whisper
{
    public class TextDecoder : nn.Module<(Tensor x, Tensor xa, Dictionary<nn.Module, Tensor>? kv_cache), Tensor>
    {
        private Embedding token_embedding;
        private Parameter positional_embedding;
        private ModuleList<ResidualAttentionBlock> blocks;
        private LayerNorm ln;
        private Tensor mask;

        public TextDecoder(int n_vocab, int n_ctx, int n_state, int n_head, int n_layer) : base(nameof(TextDecoder))
        {
            this.token_embedding = nn.Embedding(n_vocab, n_state);
            this.positional_embedding = torch.nn.Parameter(torch.empty(n_ctx, n_state));

            this.blocks = nn.ModuleList([..from _ in Enumerable.Range(0, n_layer)
                                           select new ResidualAttentionBlock(n_state, n_head, cross_attention: true)]);

            this.ln = nn.LayerNorm(n_state);

            var maskTensor = torch.empty(n_ctx, n_ctx).fill_(float.NegativeInfinity).triu_(1);
            this.mask = maskTensor;
            this.register_buffer("mask", maskTensor, persistent: false);
            this.RegisterComponents();
        }

        public ModuleList<ResidualAttentionBlock> Blocks
        {
            get => this.blocks;
            set => this.blocks = value;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input">
        /// x : torch.LongTensor, shape = (batch_size, &lt;= n_ctx)
        /// the text tokens
        /// xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
        /// the encoded audio features to be attended on.
        /// </param>
        /// <returns>logits </returns>
        public override Tensor forward((Tensor x, Tensor xa, Dictionary<nn.Module, Tensor>? kv_cache) input)
        {
            (Tensor x, Tensor xa, Dictionary<nn.Module, Tensor>? kv_cache) = input;
            var offset = kv_cache != null && kv_cache.Count > 0 ? kv_cache.Values.First().shape[1] : 0;
            x = this.token_embedding.call(x) + this.positional_embedding[(int)offset..(int)(offset + x.shape[^1])];
            x = x.to_type(xa.dtype);

            foreach (var block in this.blocks)
            {
                x = block.call((x, xa, this.mask, kv_cache));
            }

            x = this.ln.call(x);
            var logits = torch.matmul(x, this.token_embedding.weight!.transpose(0, 1).to_type(x.dtype)).@float();
            return logits;
        }
    }
}
