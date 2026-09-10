# Portfólio — Redes Neurais

Repositório de exercícios e projetos da disciplina. Cada entrega é uma pasta que
fica: o site é o portfólio, o repositório é o que foi realmente executado —
código em `code/`, figuras versionadas em `figures/`, dados em `data/`.

## Quem sou

João Gabriel — graduação em Engenharia da Computação. Interesse em como a
geometria dos dados condiciona a arquitetura que a gente escolhe, que é
justamente o fio do primeiro exercício.

## Entregas

<div class="grid cards" markdown>

- :material-scatter-plot: **[Dados](exercises/data/index.md)**

    Geometria de nuvens de pontos em 2D, não-linearidade em 5D e
    pré-processamento do Spaceship Titanic para uma rede com `tanh`.

    *Entregue*

- :material-vector-point: **Perceptron**

    Ainda não entregue.

- :material-graph-outline: **MLP**

    Ainda não entregue.

- :material-shuffle-variant: **VAE**

    Ainda não entregue.

</div>

## Como rodar o código

```bash
git clone https://github.com/Joao-Gabriel05/data-redes_neurais.git
cd data-redes_neurais
pip install -r requirements.txt

# os scripts de um exercício rodam de dentro da própria pasta code/
cd docs/exercises/data/code && python3 ex1_nuvens.py
```

Para ver o site localmente: `mkdocs serve`.
