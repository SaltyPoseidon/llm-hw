| под | конфигурация | изменения | секунд | токенов/с | ускорение к бейзлайну пода | пик памяти, GiB | val loss | шагов |
|---|---|---|---|---|---|---|---|---|
| smoke | baseline-90 | как в шаблоне | 90 | 7,075 | 1.00× | 41.2 / 43.4 | 8.133 | 110 |
| smoke | fa2-90 | attn=flash_attention_2 | 91 | 7,392 | 1.04× | 39.3 / 41.0 | 8.109 | 116 |
| mech-b | baseline-90 | как в шаблоне | 90 | 7,079 | 1.00× | 41.2 / 43.4 | 8.133 | 110 |
| mech-b | bs64-ga1-90 | per_device_train_batch_size=64, gradient_accumulation_steps=1 | — | упал (код 1) | — | — | — | — |
| mech-b | gc-90 | gradient_checkpointing=True | 90 | 5,800 | 0.82× | 17.4 / 24.0 | 8.242 | 90 |
| mech-b | offload-90 | activation_offloading=True | — | упал (код 1) | — | — | — | — |
| mech-a | baseline-90 | как в шаблоне | 90 | 7,235 | 1.00× | 41.2 / 43.4 | 8.129 | 113 |
| mech-a | padfree-fa2-90 | attn=flash_attention_2, padding_free=True | 90 | 18,885 | 2.61× | 15.1 / 21.2 | 6.914 | 299 |
| mech-a | packing-fa2-90 | attn=flash_attention_2, packing=True | 91 | 21,689 | 3.00× | 40.2 / 76.2 | 8.226 | 61 |
| mech-a | liger-90 | use_liger_kernel=True | 90 | 7,167 | 0.99× | 26.8 / 28.5 | 8.161 | 107 |
| mech-a | compile-90 | torch_compile=True | 116 | меньше 10 шагов, замера нет | — | 26.0 / 26.3 | 11.235 | 1 |
| combo-a | baseline-90 | как в шаблоне | 90 | 7,223 | 1.00× | 41.2 / 43.4 | 8.129 | 113 |
| combo-a | padfree-fa2-liger-90 | attn=flash_attention_2, padding_free=True, use_liger_kernel=True | 90 | 17,023 | 2.36× | 12.3 / 14.0 | 7.144 | 258 |
| combo-a | packing-fa2-liger-90 | attn=flash_attention_2, packing=True, use_liger_kernel=True | 91 | 25,366 | 3.51× | 25.8 / 33.1 | 8.168 | 71 |
| combo-a | padfree-fa2-liger-bs128-ga1-90 | attn=flash_attention_2, padding_free=True, use_liger_kernel=True, per_device_train_batch_size=128, gradient_accumulation_steps=1 | 90 | 22,569 | 3.12× | 20.7 / 29.2 | 7.497 | 177 |
| combo-a | packing-fa2-liger-bs64-ga1-90 | attn=flash_attention_2, packing=True, use_liger_kernel=True, per_device_train_batch_size=64, gradient_accumulation_steps=1 | 91 | 26,320 | 3.64× | 41.9 / 58.5 | 8.149 | 74 |
| combo-b | baseline-90 | как в шаблоне | 90 | 7,070 | 1.00× | 41.2 / 43.4 | 8.133 | 110 |
| combo-b | offload-padfree-fa2-90 | attn=flash_attention_2, padding_free=True, activation_offloading=True | 93 | 2,936 | 0.42× | 9.1 / 11.1 | 8.951 | 45 |
| combo-b | gc-packing-fa2-liger-bs64-ga1-90 | attn=flash_attention_2, packing=True, use_liger_kernel=True, gradient_checkpointing=True, per_device_train_batch_size=64, gradient_accumulation_steps=1 | 90 | 22,106 | 3.13× | 10.5 / 14.9 | 8.239 | 58 |
| combo-b | compile-packing-fa2-liger-240 | attn=flash_attention_2, packing=True, use_liger_kernel=True, torch_compile=True | — | упал (код 1) | — | — | — | — |
| combo-b | compile-padfree-fa2-liger-240 | attn=flash_attention_2, padding_free=True, use_liger_kernel=True, torch_compile=True | 240 | 16,497 | 2.33× | 15.3 / 16.7 | 7.156 | 259 |
| push-a | baseline-90 | как в шаблоне | 90 | 7,050 | 1.00× | 41.2 / 43.4 | 8.133 | 110 |
| push-a | packing-fa2-liger-bs96-ga1-90 | attn=flash_attention_2, packing=True, use_liger_kernel=True, per_device_train_batch_size=96, gradient_accumulation_steps=1 | 91 | 26,946 | 3.82× | 59.5 / 78.5 | 8.409 | 47 |
| push-a | packing-fa2-liger-bs128-ga1-90 | attn=flash_attention_2, packing=True, use_liger_kernel=True, per_device_train_batch_size=128, gradient_accumulation_steps=1 | — | упал (код 1) | — | — | — | — |
| push-a | packing-fa2-liger-bs64-ga1-workers4-90 | attn=flash_attention_2, packing=True, use_liger_kernel=True, per_device_train_batch_size=64, gradient_accumulation_steps=1, dataloader_num_workers=4 | 90 | 26,500 | 3.76× | 41.9 / 58.5 | 8.157 | 73 |
| push-b | baseline-90 | как в шаблоне | 90 | 7,073 | 1.00× | 41.2 / 43.4 | 8.133 | 110 |
| push-b | padfree-fa2-liger-bs256-ga1-90 | attn=flash_attention_2, padding_free=True, use_liger_kernel=True, per_device_train_batch_size=256, gradient_accumulation_steps=1 | 90 | 23,545 | 3.33× | 34.4 / 48.0 | 8.108 | 85 |
| push-b | padfree-fa2-liger-bs384-ga1-90 | attn=flash_attention_2, padding_free=True, use_liger_kernel=True, per_device_train_batch_size=384, gradient_accumulation_steps=1 | 92 | 24,425 | 3.45× | 47.3 / 69.1 | 8.210 | 64 |
| final | baseline-300 | как в шаблоне | 300 | 6,830 | 1.00× | 41.2 / 43.4 | 6.659 | 358 |
| final | packing-fa2-liger-bs64-ga1-300 | attn=flash_attention_2, packing=True, use_liger_kernel=True, per_device_train_batch_size=64, gradient_accumulation_steps=1 | 300 | 26,733 | 3.91× | 41.9 / 78.2 | 6.165 | 246 |
| final | packing-fa2-liger-bs96-ga1-300 | attn=flash_attention_2, packing=True, use_liger_kernel=True, per_device_train_batch_size=96, gradient_accumulation_steps=1 | 301 | 27,196 | 3.98× | 59.6 / 78.5 | 7.131 | 169 |
| final | padfree-fa2-liger-bs128-ga1-300 | attn=flash_attention_2, padding_free=True, use_liger_kernel=True, per_device_train_batch_size=128, gradient_accumulation_steps=1 | 300 | 22,909 | 3.35× | 21.4 / 42.0 | 5.142 | 595 |

