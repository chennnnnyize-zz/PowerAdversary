# Defense strategies from literature

## Manipulating ML: poisoning attacks and countermeasure for regression learning

### System model

The function is chosen to minimize a quadratic loss function:
$$
\mathcal{L}(\mathcal{D}_{tr}, \mathsf{\theta}) = \frac{1}{n}\sum_{i = 1}^{n} (f(\mathsf{x}_i, \mathsf{\theta}) - y_i)^2 + \lambda \Omega(\mathsf{w})
$$

### Adversarial modeling

The goal is to corrupt the learning model generated in the training phase so that the predictions on new data will be modified in the testing phase. Two setup are considered, *white-box* and *black-box* attacks. In *black-box* attacks, the attackers has no knowledge of the training set $\mathcal{D}_{tr}$ but can collect a substitute data set $\mathcal{D}_{tr}^{\prime}$. The feature set and the learning algorithm are know, while the training parameters are not. 

The *white-box* attack eventually could be modeled as 
$$
\arg \max_{\mathcal{D}_p} \;\, \mathcal{W}(\mathcal{D}^{\prime}, \mathsf{\theta}_p^{\ast}) \\
\;\,\;\,\;\,\;\, s.t. \;\, \mathsf{\theta}_p^{\ast} \in \arg \min_{\mathsf{\theta}} \mathcal{L(\mathcal{D}_{tr} \cup \mathcal{D}_{p}, \mathsf{\theta})}
$$
In the *black-box* setting, the poisoned regression parameters $\mathsf{\theta}_{p}^{\ast}$ are estimated using the substitute data.

### Attack Methods



### Comments

1. The attack model is kind of different if I understand correctly. The aim is to come up with an additional data set so that the “optimized” parameter would fail on any intact data set. 
2. And the set up is like the breakdown point, while the major difference is the evaluation. Breakdown point evaluates the parameter, but this setup evaluates the performance on “test” set.
3. This setup intrinsically attacks the fitting strategy, rather than a specific model.
4. And it uses the bi-level stackelberg game.
5. The defense strategy is still, more or less, the conventional trimmed loss. 



## The space of transferable adversarial examples

