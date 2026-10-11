# Model Card

## Tarefa

**Alvo (decidido):**

```
alvo = log1p(curtidas) − log1p(mediana de curtidas dos últimos (até) 20 Shorts do canal
                                publicados ≥ 30 dias antes do vídeo-alvo)
```

- **Curtidas, não views:** é o alvo primário do TAPI, e a forma de contar views mudou em 31/03/2025 e em 24/08/2026.
- **Relativo ao canal:** isola o efeito do conteúdo, que é o que interessa para decidir entre criativos. O baseline (mediana do canal) equivale a prever `alvo = 0`.
- **Conversão para o contrato da tool:** `predicted_metric = (1 + mediana_canal) × e^(previsão) − 1` (inversa exata do `log1p`); `relative_to_channel` é a própria previsão; os quantis do intervalo são convertidos da mesma forma.
- **Mínimo de histórico:** pelo menos 5 Shorts anteriores para entrar no treino. Abaixo disso, a tool liga o `confidence_flag`.

<!-- Complete com o que mais achar necessário (ex.: alvo secundário em faixas). -->

## Algoritmo e configuração

<!-- Modelo, hiperparâmetros principais, como o intervalo é gerado. -->

## Features

<!-- Resumo por bloco. Lista completa em FEATURES.md. -->

## Métricas

<!-- Métricas no teste: erro, ordenação, cobertura do intervalo, F1 das faixas. Comparação com o baseline. -->

## Usos pretendidos

<!-- Quem usa, para qual decisão. -->

## Usos NÃO pretendidos

<!-- O que alguém poderia tentar fazer com o modelo e não deveria. -->

## Limitações

<!-- Onde o modelo é fraco? Quando o confidence_flag deve ligar? -->
