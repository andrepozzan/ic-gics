# Cadence integration

O codigo Python nao usa arquivos CSV. O modo disponivel para treinamento e validacao usa os arquivos MAT em `dados/`.

## Modelo local com MAT

Execute na raiz da atividade:

```bash
python3 script-9-cadence.py --data-source mat
```

Esse modo carrega:

- `dados/pa_sync_extraction.mat` para treinamento;
- `dados/pa_sync_validation.mat` para validacao.

## Configuracao SSH

Os parametros do servidor ficam em `experiment_config.py`:

```python
cadence_ssh_host = 'analog2'
cadence_ssh_user = 'GRR20243424'
cadence_ssh_key = '~/.ssh/id_ed25519'
cadence_remote_directory = '~/simulation'
```

Para o servidor atual, a configuracao equivalente e:

```sshconfig
Host analog2
	Hostname 200.17.220.82
	User GRR20243424
	Port 22
	IdentityFile ~/.ssh/id_ed25519
	ControlMaster auto
	ControlPath ~/.ssh/sockets/%r@%h-%p
	ControlPersist 1h
	ForwardX11 yes
	ForwardX11Trusted yes
	Compression yes
```

O setup remoto padrao e:

```bash
source ~/cadence/gpdk045/cds
```

## Testbench Doherty

O script usa `Sim_Doherty_1.scs` e o subcircuito personalizado local
`doherty-amp.scs`. O fluxo envia ambos os arquivos para o diretorio remoto
antes de cada simulacao; o netlist exportado pelo Virtuoso no servidor nao e
usado.

O subcircuito se chama `_sub0` e usa a ordem de terminais:

```text
RF_IN RF_OUT V_B1 V_B2 V_G1 V_G2
```

O testbench usa os bias observados no netlist `SIMUL-amp`: `VDD=3.3 V`, `V_B1=V_B2=2.8 V`, `V_G1=0.65 V` e `V_G2=0.25 V`. A entrada usa `rf_scale=1.0`, pois o script ja normaliza o PWL para a faixa usada no treinamento; aplicar novamente a antiga atenuacao `0.1 * 0.5` causa forte degradacao do BER.

O fluxo usa `fc=26 GHz`, `fs=104 GHz`, `maxstep=1 ps` e
`strobeperiod=1/fs`. O sinal possui 48 subportadoras contiguas, portanto a
conversao para banda-base aplica um passa-baixas de 3 GHz depois da demodulacao
analitica. O script aplica `input_backoff_db=6 dB` tanto na extração quanto no
sinal de teste antes de gerar o PWL, reduzindo o pico de tensão por um fator
0,501187 e preservando margem para a PAPR do OFDMA. O prefixo de 1 ns usado pelo script e removido antes do alinhamento,
evitando treinar com o transiente de carga dos capacitores de bloqueio.

Além de BER e NMSE, o log da execução registra o EVM percentual e em dB. O EVM
usa os símbolos QAM transmitidos como referência e corrige previamente ganho e
rotação de fase constantes. O valor em dB usa a razão RMS (não o número
percentual), de modo que limites como `-18,1 dB` sejam comparáveis às normas.

O arquivo `ade_e.scs` auxiliar tambem e enviado automaticamente para
`~/simulation`.

## Fluxo Spectre

O fluxo Spectre executa duas simulacoes remotas:

1. gerar um estimulo de extracao;
2. executar a primeira simulacao Spectre;
3. importar `V(n_in)` e `V(n_out)` do resultado `.raw`;
4. treinar as LUTs com esses resultados;
5. executar uma segunda simulacao para validacao;
6. importar a saida real da segunda simulacao para calcular o BER.

O Python importa os traces `time`, `V(n_in)` e `V(n_out)` do resultado PSF ASCII da primeira simulacao antes de treinar as LUTs. A segunda simulacao usa o sinal predistorcido e sua saida real para a validacao.

Execute:

```bash
python3 script-9-cadence.py --run-spectre
```

## Validacao com N-port

Para testar o pipeline sem usar o Doherty transistor-level, use o modelo
comportamental local `cadence/Sim_Nport_1.scs`:

```bash
python3 script-9-cadence.py --cadence-validation
```

Esse modo envia o `Sim_Nport_1.scs` para o diretorio remoto, executa as duas
simulacoes e le `n_in`/`n_out`. O modo `--run-spectre` continua usando o
testbench Doherty e os traces configurados para ele. As duas opcoes sao
mutuamente exclusivas.

Ou informe a chave diretamente:

```bash
python3 script-9-cadence.py --run-spectre \
	--ssh-key ~/.ssh/id_ed25519
```

Uma senha em texto nao e armazenada no codigo. A chave SSH evita os prompts de senha e deve estar autorizada no servidor em `~/.ssh/authorized_keys`.

Para instalar a chave publica existente no servidor, execute uma vez no notebook:

```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub GRR20243424@200.17.220.82
```

Esse comando pede a senha uma ultima vez. Depois, teste:

```bash
ssh -i ~/.ssh/id_ed25519 GRR20243424@200.17.220.82 'which spectre'
```
