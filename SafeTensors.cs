namespace PerceptivePyro;

using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.Text.Json;
using System.Diagnostics.Contracts;

/// <summary>
/// .NET 
/// </summary>
public static class SafeTensors
{
    public static Dictionary<string, ScalarType> dtypes = new Dictionary<string, ScalarType>
    {
        ["F16"] = torch.float16,
        ["F32"] = torch.float32,
        ["I8"] = torch.int8,
        ["I16"] = torch.int16,
        ["I32"] = torch.int32,
        ["I64"] = torch.int64,
        ["U8"] = torch.uint8
    };

    /// <summary>
    /// Loads a .safetensors file into a sequence of named Tensors
    /// </summary>
    /// <param name="filename"></param>
    /// <param name="device"></param>
    /// <returns></returns>
    public static IEnumerable<(string Name, Tensor Tensor)> LoadFile(string filename, Device device)
    {
        using var fs = new FileStream(filename, FileMode.Open, FileAccess.Read);

        var maxLength = fs.Length;
        using var reader = new BinaryReader(fs);

        var metadataLength = (int)reader.ReadInt64();
        var metadataBytes = reader.ReadBytes(metadataLength);
        var metadataJson = Encoding.UTF8.GetString(metadataBytes);

        Contract.Assert(metadataJson != null, "Invalid .safetensors file, metadata is null");
        var doc = JsonDocument.Parse(metadataJson);
        var metadata = from e in doc.RootElement.EnumerateObject()
                    select (Key: e.Name, Value: e.Value.Deserialize<SafeTensorMetadata>());
        ////var metadata = JsonSerializer.Deserialize<IEnumerable<(string Name, SafeTensorMetadata Metadata)>(metadataJson);

        var result = new Dictionary<string, Tensor>();
        
        // Read them out in order, for file system cache's sake.
        foreach (var item in metadata.OrderBy(m => m.Value.data_offsets?[0] ?? 0))
        {
            if (item.Key != "__metadata__")
            {
                yield return (item.Key, CreateTensor(reader, item.Value, metadataLength + sizeof(Int64), device));
            }
        }
    }

    public static IEnumerable<(string Name, SafeTensorMetadata Metadata)> DescribeFile(string filename)
    {
        using var fs = new FileStream(filename, FileMode.Open, FileAccess.Read);

        var maxLength = fs.Length;
        using var reader = new BinaryReader(fs);

        var metadataLength = (int)reader.ReadInt64();
        var metadataBytes = reader.ReadBytes(metadataLength);
        var metadataJson = Encoding.UTF8.GetString(metadataBytes);

        Contract.Assert(metadataJson != null, "Invalid .safetensors file, metadata is null");
        var doc = JsonDocument.Parse(metadataJson);
        var metadata = from e in doc.RootElement.EnumerateObject()
                       select (Key: e.Name, Value: e.Value.Deserialize<SafeTensorMetadata>());
        return metadata.OrderBy(m => m.Value.data_offsets?[0] ?? 0);
    }

    public static Tensor CreateTensor(BinaryReader reader, SafeTensorMetadata info, long offset, Device device)
    {
        var dtype = dtypes[info.dtype];
        var shape = info.shape;
        var (start, stop) = (info.data_offsets[0], info.data_offsets[1]);

        var length = (int)(stop - start);
        reader.BaseStream.Seek(offset + start, SeekOrigin.Begin);

        Span<byte> b = reader.ReadBytes(length).AsSpan();

        // HACK: Load into an array first, doesn't avoid a copy, but just wanted to get it working. Limits tensors to 2gb or less.
        Tensor output = torch.empty(shape, dtype, torch.CPU);
        b.CopyTo(output.bytes[..b.Length]);
        return output.to(device);
    }

    /// <summary>
    /// Downloads the .safetensors for a specified model from HuggingFace into the models folder. (Must have a .safetensors model file.)
    /// </summary>
    /// <param name="model">The name of the model to download.</param>
    /// <returns>The full path to the model file.</returns>
    /// <exception cref="HttpRequestException">Http request failed.</exception>
    /// <exception cref="IOException">Disk access failed to write the file.</exception>
    public static async Task<string> DownloadWeightsAsync(string model)
    {
        var filePath = Path.GetFullPath($@".\models\{model}\model.safetensors");
        if (File.Exists(filePath))
        {
            return filePath;
        }

        Directory.CreateDirectory(Path.GetDirectoryName(filePath)!);
        $"Downloading weights from pretrained {model} to {filePath}".Dump();
        var stream = await new HttpClient().GetStreamAsync($"https://huggingface.co/{model}/resolve/main/model.safetensors");
        using var outputStream = File.OpenWrite(filePath);
        await stream.CopyToAsync(outputStream);
        return filePath;
    }

    public sealed record SafeTensorMetadata
    {
        public string dtype { get; set; }

        public long[] shape { get; set; }

        public long[] data_offsets { get; set; }
    }
}
