#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

struct scored_token {
    llama_token token;
    float logit;
};

struct dump_config {
    const char * directory;
};

static bool dump_intermediate(ggml_tensor * tensor, bool ask, void * user_data) {
    const std::string name = tensor->name;
    const bool moe = name.starts_with("ffn_moe_");
    const bool selected = name == "inp_embd" || name == "ffn_inp-0" ||
        name == "ffn_norm-0" || name.starts_with("l_out-") || moe;
    if (ask || !selected) {
        return selected;
    }
    const char * extension = tensor->type == GGML_TYPE_F32 ? ".f32" :
        tensor->type == GGML_TYPE_I32 ? ".i32" : nullptr;
    if (extension == nullptr) {
        std::fprintf(stderr, "unexpected intermediate type for %s: %s\n", tensor->name, ggml_type_name(tensor->type));
        return false;
    }
    std::vector<char> bytes(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
    const auto * config = static_cast<dump_config *>(user_data);
    std::fprintf(stderr, "dump %s type=%s shape=[%lld,%lld,%lld,%lld]\n", name.c_str(),
        ggml_type_name(tensor->type), (long long) tensor->ne[0], (long long) tensor->ne[1],
        (long long) tensor->ne[2], (long long) tensor->ne[3]);
    std::ofstream output(std::string(config->directory) + "/" + name + extension, std::ios::binary);
    output.write(bytes.data(), bytes.size());
    return output.good();
}

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
    dump_config dumps{std::getenv("LLAMA_DUMP_DIR")};
    if (dumps.directory != nullptr) {
        context_params.cb_eval = dump_intermediate;
        context_params.cb_eval_user_data = &dumps;
    }
    llama_context * context = llama_init_from_model(model, context_params);
    auto batch = llama_batch_get_one(tokens.data(), count);
    std::vector<int8_t> outputs(count, 1);
    batch.logits = outputs.data();
    if (context == nullptr || llama_decode(context, batch) != 0) {
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
