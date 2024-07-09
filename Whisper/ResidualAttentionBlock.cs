using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PerceptivePyro.Whisper
{

    public class ResidualAttentionBlock : nn.Module<(Tensor x, Tensor? xa, Tensor? mask, Dictionary<nn.Module, Tensor>? kv_cache), Tensor>
    {
        private MultiHeadAttention attn;
        private LayerNorm attn_ln;
        private MultiHeadAttention? cross_attn;
        private LayerNorm? cross_attn_ln;
        private Sequential mlp;
        private LayerNorm mlp_ln;

        public ResidualAttentionBlock(int n_state, int n_head, bool cross_attention = false) : base(nameof(ResidualAttentionBlock))
        {
            this.attn = new MultiHeadAttention(n_state, n_head);
            this.attn_ln = nn.LayerNorm(n_state);

            if (cross_attention)
            {
                this.cross_attn = new MultiHeadAttention(n_state, n_head);
                this.cross_attn_ln = nn.LayerNorm(n_state);
            }

            int n_mlp = n_state * 4;
            this.mlp = nn.Sequential(
                nn.Linear(n_state, n_mlp),
                nn.GELU(),
                nn.Linear(n_mlp, n_state)
            );
            this.mlp_ln = nn.LayerNorm(n_state);
            RegisterComponents();
        }

        public MultiHeadAttention Attn => this.attn;

        public override Tensor forward((Tensor x, Tensor? xa, Tensor? mask, Dictionary<nn.Module, Tensor>? kv_cache) input)
        {
            (Tensor x, Tensor? xa, Tensor? mask, Dictionary<nn.Module, Tensor>? kv_cache) = input;
            var attn_output = this.attn.call(x);
            var attn_norm = this.attn_ln.call(attn_output);

            if (this.cross_attn != null && this.cross_attn_ln != null)
            {
                var cross_attn_output = this.cross_attn.call(attn_norm);
                var cross_attn_norm = this.cross_attn_ln.call(cross_attn_output);
                attn_norm = cross_attn_norm; 
            }

            var mlp_output = this.mlp.call(attn_norm);
            var mlp_norm = this.mlp_ln.call(mlp_output);

            return mlp_norm; // Adjust the final output based on the actual forward logic
        }
    }
}
