#include "llama.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

struct scored_token {
    llama_token token;
    float logit;
};

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s MODEL.gguf PROMPT\n", argv[0]);
        return 2;
    }
    ggml_backend_load_all();
    auto model_params = llama_model_default_params();
    const char * gpu_layers = std::getenv("LLAMA_LOGITS_GPU_LAYERS");
    model_params.n_gpu_layers = gpu_layers == nullptr ? 0 : std::atoi(gpu_layers);
    llama_model * model = llama_model_load_from_file(argv[1], model_params);
    if (model == nullptr) {
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::string prompt = argv[2];
    const int count = -llama_tokenize(vocab, prompt.data(), prompt.size(), nullptr, 0, true, true);
    if (count <= 0) {
        llama_model_free(model);
        return 1;
    }
    std::vector<llama_token> tokens(count);
    if (llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), count, true, true) != count) {
        llama_model_free(model);
        return 1;
    }
    auto context_params = llama_context_default_params();
    context_params.n_ctx = std::max(32, count);
    context_params.n_batch = count;
    context_params.n_ubatch = count;
    context_params.no_perf = true;
    llama_context * context = llama_init_from_model(model, context_params);
    if (context == nullptr || llama_decode(context, llama_batch_get_one(tokens.data(), count)) != 0) {
        llama_free(context);
        llama_model_free(model);
        return 1;
    }
    const float * logits = llama_get_logits_ith(context, -1);
    const int vocab_size = llama_vocab_n_tokens(vocab);
    std::vector<scored_token> ranked;
    ranked.reserve(vocab_size);
    for (int token = 0; token < vocab_size; ++token) {
        ranked.push_back({token, logits[token]});
    }
    std::partial_sort(ranked.begin(), ranked.begin() + 10, ranked.end(), [](auto a, auto b) {
        return a.logit > b.logit || (a.logit == b.logit && a.token < b.token);
    });
    std::printf("tokens");
    for (const auto token : tokens) {
        std::printf(" %d", token);
    }
    std::printf("\n");
    for (int i = 0; i < 10; ++i) {
        const auto item = ranked[i];
        std::printf("%d %.9g %08x\n", item.token, item.logit, std::bit_cast<std::uint32_t>(item.logit));
    }
    llama_free(context);
    llama_model_free(model);
}
