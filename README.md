# mlpremier
Modeling Player Performance in the EPL via time-series prediction of 
Fantasy Premier League Points using traditional ML methods, GPT transfer learning, 
and 1D CNN.

# Abstract
Here we present a groundbreaking model for
forecasting English Premier League (EPL) player
performance using convolutional neural networks
(CNNs). We evaluate Ridge regression, Light-
GBM and CNNs on the task of predicting upcom-
ing player FPL score based on historical FPL data
over the previous weeks. Our baseline models,
Ridge regression and LightGBM, achieve solid
performance and emphasize the importance of
recent FPL points, influence, creativity, threat,
and playtime in predicting EPL player perfor-
mances. Our optimal CNN architecture achieves
better performance with fewer input features and
even outperforms the best previous EPL player
performance forecasting models in the literature.
The optimal CNN architecture also achieves very
strong Spearman correlation with player rankings,
indicating its strong implications for supporting
the development of FPL artificial intelligence (AI)
Agents and providing analysis for FPL managers.
We additionally perform transfer learning exper-
iments on soccer news data collected from The
Guardian, for the same task of predicting upcom-
ing player score, but do not identify a strong
predictive signal in natural language news texts,
achieving worse performance compared to both
the CNN and baseline models. Overall, our CNN-
based approach marks a significant advancement
in EPL player performance forecasting and lays
the foundation for transfer learning to other EPL
prediction tasks such as win-loss odds for sports
betting and the development of cutting-edge FPL
AI Agents.

[Data scraped from `vaastav`'s FPL Data Repository](https://github.com/vaastav/Fantasy-Premier-League)
