using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Amazon.Bedrock.Model;
using Bookstore.Data.BedrockService;
using Bookstore.Domain;
using Microsoft.AspNetCore.Mvc;

namespace Bookstore.Web.Areas.Admin.Controllers.Api;

[Route("api/[controller]")]
[ApiController]
public class BedrockController : AdminAreaControllerBase
{
    private readonly IBedrockService bedrockService;
    private readonly IImageResizeService imageResizeService;

    public BedrockController(
        IBedrockService bedrockService,
        IImageResizeService imageResizeService)
    {
        this.bedrockService = bedrockService;
        this.imageResizeService = imageResizeService;
    }

    [HttpGet]
    public Task<IEnumerable<FoundationModelSummary>> Get([FromQuery] string[] inputModalities)
    {
        var models = bedrockService.ListFoundationModelsAsync(inputModalities);
        return models;
    }

    [HttpPost("text")]
    public async Task<string> PostText([FromBody] TextInput input)
    {
        var response = await bedrockService.GenerateTextAsync(input.ModelId, input.Prompt);
        var content = response.Messages.ToArray()[1].Content;

        return content;
    }

    [HttpPost("image")]
    public async Task<List<string>> PostImage([FromBody] ImageInput input)
    {
        var response = await bedrockService.GenerateImageAsync(input.ModelId, input.Prompt, input.NumOfImages);

        var arrBase64 = new List<string>();
        foreach (var image in response.Images)
        {
            var stream = await imageResizeService.ResizeImageAsync(image.ToStream());
            var data = LangChain.Providers.Data.FromStream(stream);
            arrBase64.Add(data.ToBase64());
        }

        return arrBase64;
    }
}

public class ImageInput : TextInput
{
    public int NumOfImages { get; set; }
}

public class TextInput
{
    public string ModelId { get; set; }
    public string Prompt { get; set; }
}