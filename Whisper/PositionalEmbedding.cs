
namespace PerceptivePyro.Whisper
{
    using System;
    using System.Diagnostics.Contracts;

    internal static class PositionalEmbedding
    {
        /// <summary>
        /// Calculates sinusoids for positional embedding.
        /// </summary>
        /// <param name="length">Sequence length</param>
        /// <param name="channels">Embedding channels.</param>
        /// <param name="max_timescale">Maximum time scale.</param>
        /// <returns>Tensor of sinusoidal positional embeddings.</returns>
        public static Tensor Sinusoids(int length, int channels, int max_timescale = 10000)
        {
            Contract.Assert(channels % 2 == 0);
            var log_timescale_increment = torch.log(torch.tensor(max_timescale)).item<float>() / (channels / 2 - 1);

            // Calculate inv_timescales
            var invTimescales = torch.exp(-log_timescale_increment * torch.arange(channels / 2));

            // Calculate scaled_time
            var scaledTime = torch.arange(length).unsqueeze(1) * invTimescales.unsqueeze(0);

            // Concatenate sin and cos of scaled_time
            return torch.cat([torch.sin(scaledTime), torch.cos(scaledTime)], 1);
        }
    }
}
