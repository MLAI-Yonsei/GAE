# 의료 실험 결과

현재 GAE의 주 지표와 선택 기준은 nAOPC·nComp(M) 두 가지이며 ΔCE는 과거 참고값으로만 남긴다.
두 지표 실행 계획에 선택 기준과 실행 근거를 기록한다.
새 TC 결과는 기존 decoder를 현재 validation 규약으로 재선택하며 SAE는 동일 adaptation 입력에서 decoder를 새로 fitting한다.
ID 모델·encoder는 고정하고 adaptation·validation·evaluation 분할을 유지하며 최종 평가 점수는 선택에 사용하지 않는다.
위 재선택 설명은 기존 GAE 기록의 조건이며, GAE+ V2는 adaptation S에서 HP·혼합비를 선택하고 별도 평가를 수행한 조건을 따른다.

## 주 비교표

### GPT-2 Small + Transcoder

| 방법          | PubMedQA (PQA-U) nAOPC↑ | PubMedQA (PQA-U) nComp(M)↑ | PMC-Patients nAOPC↑ | PMC-Patients nComp(M)↑ | MedQuAD nAOPC↑ | MedQuAD nComp(M)↑ |
| ----------- | ----------------------: | -------------------------: | ------------------: | ---------------------: | -------------: | ----------------: |
| Fixed       | 0.8796                  | 1.1194                     | 0.8702              | 1.1064                 | 0.8742         | 1.0720            |
| TERM        | —                       | —                          | —                   | —                      | —              | —                 |
| Finetune    | 0.8489                  | 1.0415                     | 0.8362              | 1.0073                 | 0.8236         | 0.9589            |
| Retrain     | 0.8700                  | 1.0500                     | 0.8729              | 1.0383                 | 0.8178         | 0.9201            |
| SAEBoost    | 0.8842                  | 1.1491                     | 0.8795              | 1.1452                 | 0.9000         | 1.1230            |
| FaithfulSAE | 0.9082                  | 1.2750                     | 0.9002              | 1.2646                 | 0.8882         | 1.1357            |
| GAE         | ++0.9172++              | ++1.2912++                 | ++0.9091++          | ++1.2680++             | ++0.8937++     | ++1.2016++        |
| GAE+        | **0.9251**              | **1.3853**                 | **0.9203**          | **1.2942**             | **0.9041**     | **1.2511**        |

GAE+ / pubmedqa_pqa_u: 설정·선택·원시 결과 (실행 `20261110T011125Z_medical_plus_remote`)이며 N=1024이다.

GAE+ / pmc_patients_v1: 설정·선택·원시 결과 (실행 `20261110T011125Z_medical_plus_remote`)이며 N=1024이다.

GPT-2 TC / PMC-Patients의 선택 decoder family는 `original`이며, 현재 수정 GAE+의 후보 선택·Fixed fallback 규약으로 평가한 값이다.

GAE+ / medquad: 설정·선택·원시 결과 (실행 `20261110T011125Z_medical_plus_remote`)이며 N=1024이다.

### Pythia-1.4B + Transcoder

| 방법          | PubMedQA (PQA-U) nAOPC↑ | PubMedQA (PQA-U) nComp(M)↑ | PMC-Patients nAOPC↑ | PMC-Patients nComp(M)↑ | MedQuAD nAOPC↑ | MedQuAD nComp(M)↑ |
| ----------- | ----------------------: | -------------------------: | ------------------: | ---------------------: | -------------: | ----------------: |
| Fixed       | 0.7275                  | 0.8272                     | 0.7072              | 0.8393                 | 0.7132         | 0.8301            |
| TERM        | —                       | —                          | —                   | —                      | —              | —                 |
| Finetune    | 0.7232                  | 0.8266                     | 0.6887              | 0.7975                 | 0.7177         | 0.8419            |
| Retrain     | 0.7113                  | 0.8010                     | 0.6647              | 0.7278                 | 0.6760         | 0.7712            |
| SAEBoost    | 0.7386                  | 0.8862                     | 0.7392              | 0.9017                 | ++0.7452++     | ++0.8896++        |
| FaithfulSAE | 0.7269                  | 0.8356                     | 0.6899              | 0.7978                 | 0.7104         | 0.8260            |
| GAE         | ++0.7874++              | ++0.9802++                 | ++0.7673++          | ++0.9654++             | 0.7449         | 0.8819            |
| GAE+        | **0.8049**              | **1.0675**                 | **0.7748**          | **1.0298**             | **0.7530**     | **0.9192**        |

Pythia-1.4B TC / pubmedqa_pqa_u의 선택 decoder family는 `original`이며, 현재 수정 GAE+의 후보 선택·Fixed fallback 규약으로 평가한 값이다.

GAE+ / pubmedqa_pqa_u: 설정·선택·원시 결과 (실행 `20261110T011125Z_medical_plus_remote`)이며 N=2048이다.

Pythia-1.4B TC / pmc_patients_v1의 선택 decoder family는 `original`이며, 현재 수정 GAE+의 후보 선택·Fixed fallback 규약으로 평가한 값이다.

GAE+ / pmc_patients_v1: 설정·선택·원시 결과 (실행 `20261110T011125Z_medical_plus_remote`)이며 N=2048이다.

Pythia-1.4B TC / MedQuAD의 선택 decoder family는 `original`이며, 현재 수정 GAE+의 후보 선택·Fixed fallback 규약으로 평가한 값이다.

GAE+ / medquad: 설정·선택·원시 결과 (실행 `20261110T011125Z_medical_plus_remote`)이며 N=1024이다.

### GPT-2 Small + SAE

| 방법          | PubMedQA (PQA-U) nAOPC↑ | PubMedQA (PQA-U) nComp(M)↑ | PMC-Patients nAOPC↑ | PMC-Patients nComp(M)↑ | MedQuAD nAOPC↑ | MedQuAD nComp(M)↑ |
| ----------- | ----------------------: | -------------------------: | ------------------: | ---------------------: | -------------: | ----------------: |
| Fixed       | 0.7386                  | 0.8468                     | 0.7384              | 0.8301                 | 0.7351         | 0.8023            |
| TERM        | —                       | —                          | —                   | —                      | —              | —                 |
| Finetune    | 0.7318                  | 0.8161                     | 0.7318              | 0.8133                 | 0.7274         | 0.7838            |
| Retrain     | 0.7647                  | 0.8628                     | 0.7566              | 0.8699                 | 0.7462         | 0.8214            |
| SAEBoost    | 0.7534                  | 0.8582                     | 0.7594              | 0.8853                 | 0.7618         | 0.8382            |
| FaithfulSAE | 0.7521                  | 0.8583                     | 0.7493              | 0.8671                 | 0.7606         | 0.8505            |
| GAE         | ++0.7923++              | ++0.8826++                 | ++0.7910++          | **0.9109**             | ++0.7779++     | ++0.8718++        |
| GAE+        | **0.8086**              | **0.9040**                 | **0.8018**          | ++0.9043++             | **0.8017**     | **1.0629**        |

GAE+ / pubmedqa_pqa_u: 설정·선택·원시 결과 (실행 `20261110T011125Z_medical_plus_remote`)이며 N=1024이다.

GAE+ / pmc_patients_v1: 설정·선택·원시 결과 (실행 `20261110T011125Z_medical_plus_remote`)이며 N=1024이다.

GAE+ / medquad: 설정·선택·원시 결과 (실행 `20261110T011125Z_medical_plus_remote`)이며 N=128이다.

### Pythia-1.4B + SAE

| 방법          | PubMedQA (PQA-U) nAOPC↑ | PubMedQA (PQA-U) nComp(M)↑ | PMC-Patients nAOPC↑ | PMC-Patients nComp(M)↑ | MedQuAD nAOPC↑ | MedQuAD nComp(M)↑ |
| ----------- | ----------------------: | -------------------------: | ------------------: | ---------------------: | -------------: | ----------------: |
| Fixed       | 0.9565                  | 1.6451                     | ++0.9225++          | 1.3838                 | 0.9479         | 1.5260            |
| TERM        | —                       | —                          | —                   | —                      | —              | —                 |
| Finetune    | 0.9498                  | 1.5680                     | 0.9237              | 1.4076                 | 0.9532         | 1.4937            |
| Retrain     | 0.9546                  | 1.4972                     | 0.9212              | 1.3883                 | 0.9354         | 1.4153            |
| SAEBoost    | 0.9548                  | 1.6272                     | ++0.9291++          | 1.3682                 | 0.9456         | 1.4756            |
| FaithfulSAE | 0.9514                  | 1.5624                     | 0.9204              | 1.4013                 | 0.9583         | ++1.5764++        |
| GAE         | **0.9579**              | **1.6697**                 | 0.9280              | ++1.4138++             | **0.9602**     | 1.5554            |
| GAE+        | ++0.9539++              | ++1.6575++                 | **0.9464**          | **1.6528**             | ++0.9589++     | **1.6321**        |

GAE+ / pubmedqa_pqa_u: 설정·선택·원시 결과 (실행 `20261111T064840Z_medical_sae_server79`)이며 N=8192이다.

GAE+ / pmc_patients_v1: 설정·선택·원시 결과 (실행 `20261111T064840Z_medical_sae_server79`)이며 N=2048이다.

GAE+ / medquad: 설정·선택·원시 결과 (실행 `20261111T064840Z_medical_sae_server79`)이며 N=2048이다.

Pythia-1.4B + SAE / MedQuAD: 설정·선택 decoder·원시 결과 (실행 `20261109T161109Z_medquad_campaign_00026`).
