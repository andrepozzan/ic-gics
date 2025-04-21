<p align="center">
  <img src="./assets/gics-logo.png" alt="GICS Logo" width="200" />
</p>

# 📂 Repositório de Códigos da Iniciação Científica (GICS)

> Este repositório agrupa todos os scripts e implementações computacionais desenvolvidos ao longo da minha **Iniciação Científica** no grupo GICS. Aqui você encontrará códigos em Python e MATLAB, organizados de acordo com cada atividade/reporte técnico.

---

## 📑 Sumário

- [📖 Sobre](#-sobre)
- [🗂 Estrutura do Repositório](#-estrutura-do-repositório)
- [⚙️ Requisitos](#️-requisitos)
- [🚀 Instalação & Uso](#-instalação--uso)
- [📄 Relatórios Vinculados](#-relatórios-vinculados)
- [🔗 Referências Técnicas](#-referências-técnicas)
- [👤 Autor](#-autor)
- [⚖️ Licença](#️-licença)

---

## 📖 Sobre

Este repositório foi criado para:

- **Armazenar** todos os códigos utilizados nos relatórios de Atividades da IC.
- **Facilitar** a reprodução dos experimentos e simulações.
- **Documentar** o uso de métodos de mínimos quadrados, séries de Volterra e modelos de Polinômio com Memória (MP).

---

## 🗂 Estrutura do Repositório

```text
/
├── Atividade_1_IC_GICS/        # Ajuste de reta por Mínimos Quadrados
│   ├── main.py                 # Exemplo de uso
│   ├── least_squares.py        # Implementação em Python (NumPy + Matplotlib)
│   └── report_figures/         # Gráficos comparativos
│
├── Atividade_2_IC_GICS/        # Modelagem via série de Volterra (amplificador)
│   ├── main.py
│   ├── volterra.py             # Geração da matriz de regressão
│   └── figures/
│
├── Atividade_3_IC_GICS/
│   ├── main.py                 # Modelo MP com sinais complexos
│   ├── memory_polynomial.py    # Cálculo de coeficientes (np.linalg.lstsq)
│   └── figures/                # AM‑AM scatter plots, NMSE
│
├── assets/
│   └── gics_logo.png           # Logo do GICS (utilizado neste README)
│
├── requirements.txt            # Bibliotecas Python
└── README.md                   # Este arquivo
```
