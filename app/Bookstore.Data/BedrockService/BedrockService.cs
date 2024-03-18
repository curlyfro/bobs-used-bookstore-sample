#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Amazon.Bedrock;
using Amazon.Bedrock.Model;
using LangChain.Providers;
using LangChain.Providers.Amazon.Bedrock;

namespace Bookstore.Data.BedrockService;

public interface IBedrockService
{
    Task<IEnumerable<FoundationModelSummary>?> ListFoundationModelsAsync(string[] outputModalities);
    Task<ChatResponse> GenerateTextAsync(string modelId, string prompt, BinaryData? imageData = null);
    Task<TextToImageResponse> GenerateImageAsync(string modelId, string prompt, int numOfImages = 1);
}

public class BedrockService : IBedrockService
{
    private readonly IAmazonBedrock bedrockClient;

    public BedrockService(IAmazonBedrock bedrockClient)
    {
        this.bedrockClient = bedrockClient;
    }

    public async Task<IEnumerable<FoundationModelSummary>?> ListFoundationModelsAsync(string[] outputModalities)
    {
        var allModels = (await bedrockClient.ListFoundationModelsAsync(new ListFoundationModelsRequest())).ModelSummaries
            .Where(x => x.OutputModalities.Intersect(outputModalities).Any())
            .OrderBy(x => x.ProviderName);
        var foundationModels = Constants.ListValidModels(allModels);

        return foundationModels;
    }

    public Task<ChatResponse> GenerateTextAsync(string modelId, string prompt, BinaryData? imageData = null)
    {
        modelId = modelId ?? throw new ArgumentNullException(nameof(modelId));

        var chatRequest = ChatRequest.ToChatRequest(prompt);
        chatRequest.Image = imageData;

        var llm = Constants.GetModelTypeById<ChatModel>(modelId);
        var response = llm.GenerateAsync(chatRequest);

        return response;
    }

    public Task<TextToImageResponse> GenerateImageAsync(string modelId, string prompt, int numOfImages = 1)
    {
        modelId = modelId ?? throw new ArgumentNullException(nameof(modelId));

        var rand = new Random();

        var llm = Constants.GetModelTypeById<ITextToImageModel>(modelId);
        var response = llm.GenerateImageAsync(prompt, new BedrockImageSettings { Seed = rand.Next(), NumOfImages = numOfImages, });

        return response;
    }
}