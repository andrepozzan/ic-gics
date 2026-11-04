# Atividade 10 — Cadence, RFWebLab e MAT

O script principal da atividade é:

```text
python/script-10-cadence-rfweblab.py
```

Escolha o fluxo correspondente:

- [Cadence/Spectre](cadence/README.md): simulação remota, SSH, testbench
  Doherty e validação N-port;
- [RFWebLab](python/rfweblab/README.md): medição remota, protocolo HTTP, limites de
  segurança, polling e gráficos;
- [Arquivos MAT locais](mat/README.md): treinamento e validação usando os
  arquivos existentes em `dados/`.

Todos os fluxos salvam os resultados em:

```text
resultados/AAAAmmdd_HHMMSS/
```

Essa pasta é ignorada pelo Git. Cada execução contém o `execution.log`, os
gráficos produzidos e `pa_analysis.mat` para auditoria independente.
