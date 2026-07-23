# mpc-metodonewton-l2c
Repositorio de o TCC do curso de cálculo númerico da L2C - Prof Rafael Gontijo
# Controle Preditivo (MPC) com Otimização por Método de Newton

TCC — Curso de Cálculo Numérico, Turma 2 (L2C)  
Autor: Frederico Augusto Montandon Lima

## Descrição do Problema

Controlar a temperatura de um processo térmico de primeira ordem, levando-a de 20 °C
até a referência de 50 °C, usando **MPC (Model Predictive Control)**.

O processo é descrito em tempo discreto por:

```
T[k+1] = a·T[k] + b·u[k]        a = 0.9   b = 0.1
```

onde `T` é a temperatura (CV) e `u` é a potência aplicada (MV, limitada a 0–100%).

A cada instante de amostragem o controlador resolve um problema de otimização:
encontrar a sequência de N ações de controle que minimiza a função custo

```
J(u) = Σ_{k=0}^{N-1} [ (T[k] − T_ref)² + λ_u · u[k]² ]
```

O primeiro termo penaliza o erro em relação à referência; o segundo penaliza o
esforço de controle (λ_u = 0.1), evitando ações agressivas. Aplica-se apenas
`u[0]` na planta e repete-se o cálculo no próximo passo — o princípio do
**horizonte deslizante**.

O interesse numérico do problema está em **como** minimizar J: aqui isso é feito
pelo **método de Newton**, com gradiente e hessiana obtidos por **diferenças
finitas**..

## Explicação do Código

### Parâmetros
| Símbolo | Valor | Significado |
|---|---|---|
| `a`, `b` | 0.9, 0.1 | coeficientes do modelo discreto |
| `T_ref` | 50 | setpoint de temperatura |
| `N` | 10 | horizonte de predição/controle |
| `lambda_u` | 0.1 | peso da penalização do controle |
| `steps` | 50 | passos de simulação em malha fechada |

### `funcao_custo(u_seq, T0)`
Simula o modelo N passos à frente a partir de `T0`, acumulando erro quadrático e
esforço de controle. Retorna o escalar J. É a função objetivo a ser minimizada.

### `gradiente(u_seq, T0)`
Aproxima ∂J/∂uᵢ por **diferenças finitas progressivas**:

```
∂J/∂uᵢ ≈ [ J(u + ε·eᵢ) − J(u) ] / ε        ε = 1e-3
```

Custo: N+1 avaliações de J.

### `hessiana(u_seq, T0)`
Aproxima as derivadas segundas cruzadas por diferenças finitas de segunda ordem:

```
∂²J/∂uᵢ∂uⱼ ≈ [ J(u+εeᵢ+εeⱼ) − J(u+εeᵢ) − J(u+εeⱼ) + J(u) ] / ε²   ε = 1e-4
```

Resulta na matriz H (N×N), simétrica e positiva definida — J é quadrática em u,
o que garante convergência do método de Newton.

### `newton_optimizer(T0)`
Implementa a iteração de Newton multivariável:

```
H · δ = ∇J          →      u ← u − δ
```

Detalhes:
- `np.linalg.solve(H, grad)` resolve o sistema linear em vez de inverter H
  (mais estável e barato numericamente);
- `np.clip(u_seq, 0, 100)` impõe as restrições físicas do atuador;
- para em `‖δ‖ < 1e-6` ou após 15 iterações.

Como J é quadrática, o método converge em **uma iteração** na ausência de
saturação — as demais servem para acomodar o `clip`.

### Laço de simulação
Para cada instante k: otimiza a sequência a partir da temperatura atual, aplica
somente `u_opt[0]`, propaga o modelo e descarta o restante da sequência
(horizonte deslizante).

### Gráficos
Dois subplots: temperatura com a referência tracejada, e o sinal de controle ao
longo do tempo.

## Como executar

pip install numpy matplotlib
python mpc_newton.py

```
