# Fluxo Cadence/Spectre

O modo Cadence executa o testbench Spectre local ou remoto e usa os traces
`V(n_in)` e `V(n_out)` para treinar e validar o modelo.

## Execucao

Validacao com o N-port comportamental:

```bash
python3 python/script-10-cadence-rfweblab.py --cadence-validation
```

Simulacao transistor-level Doherty:

```bash
python3 python/script-10-cadence-rfweblab.py --run-spectre
```

Também é possível selecionar explicitamente a fonte:

```bash
python3 python/script-10-cadence-rfweblab.py --data-source cadence
```

Esse modo requer SSH configurado para o servidor Cadence. Os argumentos
disponíveis podem ser consultados com:

```bash
python3 python/script-10-cadence-rfweblab.py --help
```

Opções úteis:

```bash
--ssh-host HOST
--ssh-user USER
--ssh-key ~/.ssh/id_ed25519
--remote-directory ~/simulation
--remote-setup-command "source ~/cadence/gpdk045/cds"
```

Nenhuma senha ou chave privada é armazenada no código.

## Arquivos Cadence

O fluxo usa:

- `cadence/Sim_Doherty_1.scs`;
- `cadence/Sim_Nport_1.scs`;
- `cadence/doherty-amp.scs`;
- `cadence/ade_e.scs`.

O subcircuito Doherty usa os terminais:

```text
RF_IN RF_OUT V_B1 V_B2 V_G1 V_G2
```

Os arquivos de estímulo PWL e os resultados das execuções são temporários e
não devem ser versionados.

## Parametros de simulacao

O fluxo usa:

```text
fc = 26 GHz
fs = 104 GSa/s
maxstep = 1 ps
strobeperiod = 1/fs
```

O sinal OFDMA usa 48 subportadoras contíguas. Após a demodulação, a conversão
para banda-base aplica um filtro passa-baixas de 3 GHz.

O sinal recebe back-off de entrada de 6 dB, e o prefixo de acomodação de 1 ns
é removido antes do alinhamento e do treinamento.

## Resultados

Os resultados são salvos em:

```text
resultados/AAAAmmdd_HHMMSS/
```

O fluxo exporta `pa_analysis.mat`, métricas, gráficos, sinais intermediários e
o log completo da execução.
