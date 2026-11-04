# Fluxo RFWebLab

O cliente RFWebLab fica isolado nesta pasta e implementa o protocolo do arquivo
[`RFWebLab_PA_meas_v1_2.m`](../../RFWebLab_PA_meas_v1_2.m).

## Execucao

Na raiz da atividade:

```bash
python3 python/script-10-cadence-rfweblab.py --data-source rfweblab
```

A potencia RMS de entrada pode ser informada em dBm:

```bash
python3 python/script-10-cadence-rfweblab.py \
    --data-source rfweblab \
    --rfweblab-rmsin -20
```

O fluxo gera `rfweblab_communication.png`, contendo a entrada, a saída e a
resposta entrada-saída do amplificador remoto.

## Comunicacao e polling

O processo e:

1. serializar o cluster MATLAB;
2. calcular o checksum MD5;
3. fazer POST para `upload.php`;
4. receber `output_<id>.dat`;
5. consultar `ok_<id>.dat` ate a medicao terminar;
6. baixar `output_<id>.zip`;
7. extrair e desserializar `output_<id>.dat`.

O arquivo `ok_<id>.dat` e somente um marcador de conclusao. O arquivo real
contendo a resposta esta no ZIP:

```text
https://dpdcompetition.com/rfweblab/matlab/files/output_<id>.zip
```

O `.dat` final pode ser consultado em:

```text
https://dpdcompetition.com/rfweblab/matlab/files/output_<id>.dat
```

Respostas HTTP 400 e 404 durante o polling sao esperadas enquanto o servidor
processa a medicao. O intervalo padrao e 10 segundos.

## Limites e seguranca

O cliente rejeita:

- sinais com `NaN` ou `Inf`;
- menos de 1.000 amostras;
- PAPR acima de 20 dB;
- potencia de pico acima de -8 dBm.

O RFWebLab aceita no maximo 1.000.000 amostras, opera a 200 MHz e possui banda
util aproximada de -80 MHz a +80 MHz. O cliente usa HTTPS por padrao e nao
armazena senhas ou chaves.

Os endpoints podem ser alterados por ambiente:

```bash
export RFWEBLAB_UPLOAD_URL="https://..."
export RFWEBLAB_FILES_URL="https://..."
```

## Configuracao

As constantes principais ficam no inicio de
[`client.py`](client.py):

```python
SAMPLE_RATE = 200e6
MAX_SAMPLES = 1_000_000
MIN_SAMPLES = 1_000
REQUEST_TIMEOUT_SECONDS = 30.0
POLL_INTERVAL_SECONDS = 10.0
MAX_POLL_ATTEMPTS = 100
```

Tambem podem ser configuradas sem editar o arquivo:

```bash
export RFWEBLAB_TIMEOUT_SECONDS=30
export RFWEBLAB_POLL_INTERVAL_SECONDS=10
export RFWEBLAB_MAX_POLL_ATTEMPTS=100
```

O resultado remoto e alinhado por atraso fracionario e ganho complexo antes do
treinamento das LUTs. Esse alinhamento e necessario porque a captura do
RFWebLab pode iniciar em uma posição temporal diferente da entrada.

## Logs

Cada execucao salva:

```text
resultados/AAAAmmdd_HHMMSS/execution.log
```

O terminal usa cores por nivel. O arquivo permanece sem codigos ANSI. Defina
`NO_COLOR=1` para desativar as cores.
