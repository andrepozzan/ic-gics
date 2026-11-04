mat# Fluxo com arquivos MAT locais

O modo MAT usa os arquivos de dados já existentes na atividade e não faz
upload para o RFWebLab.

## Execucao

Na raiz da atividade:

```bash
python3 python/script-10-cadence-rfweblab.py --data-source mat
```

Os arquivos utilizados são:

```text
dados/pa_sync_extraction.mat
dados/pa_sync_validation.mat
```

Cada arquivo deve conter os campos `x` e `y`, correspondentes à entrada e à
saída complexas do amplificador.

## Processamento

O fluxo:

1. carrega os dois arquivos MAT;
2. alinha atraso entre entrada e saída;
3. compensa ganho complexo;
4. treina o modelo LUT;
5. treina a pré-distorção;
6. calcula BER, EVM e NMSE;
7. salva sinais, métricas e gráficos.

O alinhamento é registrado no log como:

```text
MAT alignment: delay ... samples; gain (...)
```

## Resultados

Os resultados ficam em:

```text
resultados/AAAAmmdd_HHMMSS/
```

Entre os arquivos gerados estão:

- `pa_analysis.mat`;
- `execution.log`;
- `training_am_am.png`;
- `validation_real_comparison.png`;
- `psd_comparison.png`;
- `time_domain_comparison.png`;
- gráficos de constelação;
- arquivos CSV auxiliares dos gráficos.

## Auditoria Octave/MATLAB

O Python não executa o Octave automaticamente. Para auditar a execução mais
recente sem abrir janelas:

```bash
octave --no-gui --quiet --path octave \
    --eval "auditar_pa('', false)"
```

Para auditar uma execução específica:

```bash
octave --no-gui --quiet --path octave \
    --eval "auditar_pa('resultados/AAAAmmdd_HHMMSS', false)"
```

O relatório é salvo em:

```text
resultados/AAAAmmdd_HHMMSS/octave_report/
```

No MATLAB:

```matlab
addpath('octave');
auditar_pa('resultados/AAAAmmdd_HHMMSS', true);
```
