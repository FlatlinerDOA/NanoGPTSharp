namespace PerceptivePyro;

using System;
using System.Numerics;

public static class TensorExtensions
{
    public static void set_seed(int seed)
    {
        manual_seed(seed);
        cuda.manual_seed(seed);
        random.manual_seed(seed);
        backends.cuda.matmul.allow_tf32 = true; // allow tf32 on matmul
        backends.cudnn.allow_tf32 = true; // allow tf32 on cudnn
    }


    public static IEnumerable<List<T>> ToLists<T>(this Tensor tensor) where T : unmanaged
    {
        for (int y = 0; y < tensor.shape[0]; y++)
        {
            yield return tensor[y, ..].data<T>().ToList<T>();
        }
    }

    public static T[,] ToMultiDimensional2d<T>(this Tensor tensor) where T : unmanaged
    {
        var array = new T[tensor.shape[0], tensor.shape[1]];
        for (int y = 0; y < tensor.shape[0]; y++)
        {
            for (var x = 0; x < tensor.shape[1]; x++)
            {
                array[y, x] = tensor[y, x].item<T>();
            }
        }

        return array;
    }


    public static T[,] ToMultiDimensional<T>(this IReadOnlyList<T[]> arrays)
    {
        var array = new T[arrays.Count,arrays.Select(a => a.Length).Max()];
        for (int y = 0; y < arrays.Count; y++)
        {
            var a = arrays[y];            
            for (var i = 0; i < a.Length; i++)
            {
                array[y,i] = a[i];
            }
        }

        return array;
    }

    public static ScalarType ToTensorType(this Type type) => type.Name switch
    {
        nameof(Boolean) => ScalarType.Bool,
        nameof(Byte) => ScalarType.Int8,
        nameof(Int32) => ScalarType.Int32,
        nameof(Int64) => ScalarType.Int64,
        nameof(Half) => ScalarType.BFloat16,
        nameof(Single) => ScalarType.Float32,
        nameof(Double) => ScalarType.Float64,
        _ => throw new NotImplementedException()
    };

    public static Tensor to_sparse(this Tensor tensor) => sparse_coo_tensor(tensor.SparseIndices, tensor.SparseValues, tensor.shape, ScalarType.Bool);

    public static Tensor normalize(this Tensor input, float p = 2.0f, int dim = 1, float eps = 1e-12f)
    {
        var denom = input.norm(dim: dim, keepdim: true, p: p).clamp_min(eps).expand_as(input);
        return input / denom;
    }

    /// <summary>
    /// Useful for calculating embeddings, by taking token embeddings and calculating their mean values
    /// Shape input is (Batch, Time, Embedding Channel) to output (Batch, Embedding Channel).
    /// </summary>
    /// <param name="model_output"></param>
    /// <param name="attention_mask"></param>
    /// <returns>Tensor of shape (Batch, Embedding Channel)</returns>
    public static Tensor mean_pooling(this Tensor model_output, Tensor attention_mask)
    {
        var token_embeddings = model_output; // token embeddings
        var input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).@float();
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min: 1e-9);
    }

    /// <summary>
    /// Splits the tensor as per Pytorch's iter() implementation for tensors.
    /// >>> [s for s in torch.tensor([[0, 1], [2, 3]])]
    /// [tensor([0, 1]), tensor([2, 3])]
    /// </summary>
    /// <param name="tensor"></param>
    /// <returns></returns>
    public static IEnumerable<Tensor> AsEnumerable(this Tensor tensor)
    {
        for (int i = 0; i < tensor.shape[0]; i++)
        {
            yield return tensor[i];
        }
    }

    public static IEnumerable<T> AsEnumerable<T>(this ValueTuple<T> input) => new[] { input.Item1 };
    public static IEnumerable<T> AsEnumerable<T>(this ValueTuple<T, T> input) => new[] { input.Item1, input.Item2 };
    public static IEnumerable<T> AsEnumerable<T>(this ValueTuple<T, T, T> input) => new[] { input.Item1, input.Item2, input.Item3 };
    public static IEnumerable<T> AsEnumerable<T>(this ValueTuple<T, T, T, T> input) => new[] { input.Item1, input.Item2, input.Item3, input.Item4 };
    public static IEnumerable<T> AsEnumerable<T>(this ValueTuple<T, T, T, T, T> input) => new[] { input.Item1, input.Item2, input.Item3, input.Item4, input.Item5 };
}
