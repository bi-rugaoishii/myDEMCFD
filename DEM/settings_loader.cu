#include <stdio.h>
#include <stdlib.h>
#include "cJSON_d.h"
#include "settings_loader.h"
#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic pop

cJSON* load_json_file(const char* filename){
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        printf("File open error\n");
        return NULL;
    }

    // ファイルサイズ取得
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    rewind(fp);

    // バッファ確保
    char* buffer = (char*)malloc(size + 1);
    if (!buffer) {
        fclose(fp);
        return NULL;
    }

    // 読み込み
    size_t read_bytes = fread(buffer, 1, size, fp);
    buffer[size] = '\0';
    fclose(fp);

    // JSONパース
    cJSON* root = cJSON_Parse(buffer);

    if (!root) {
        printf("Parse error: %s\n", cJSON_GetErrorPtr());
    }

    free(buffer);  // ← パース後は不要

    return root;   // ← ここがポイント
}
