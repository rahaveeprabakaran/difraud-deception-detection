# Cross-domain transfer (one BERT-base run per source domain, seed 42, fixed settings)

Rows = training domain, columns = test domain. Diagonal (bold) = in-domain test. AUC does not depend on a threshold or on how common deception is; 0.5 is chance.

## ROC AUC

| Train on \ Test on | fake_news | job_scams | phishing | political_statements | product_reviews | sms | twitter_rumours |
|---|---|---|---|---|---|---|---|
| fake_news | **0.998** | 0.628 | 0.629 | 0.543 | 0.537 | 0.818 | 0.525 |
| job_scams | 0.338 | **0.963** | 0.758 | 0.462 | 0.516 | 0.199 | 0.501 |
| phishing | 0.314 | 0.562 | **0.999** | 0.451 | 0.562 | 0.888 | 0.459 |
| political_statements | 0.772 | 0.531 | 0.427 | **0.664** | 0.532 | 0.177 | 0.512 |
| product_reviews | 0.566 | 0.384 | 0.752 | 0.490 | **0.784** | 0.574 | 0.421 |
| sms | 0.685 | 0.486 | 0.677 | 0.492 | 0.499 | **1.000** | 0.510 |
| twitter_rumours | 0.375 | 0.636 | 0.664 | 0.500 | 0.382 | 0.504 | **0.931** |


## Binary F1 at threshold 0.5 (deceptive = positive)

| Train on \ Test on | fake_news | job_scams | phishing | political_statements | product_reviews | sms | twitter_rumours |
|---|---|---|---|---|---|---|---|
| fake_news | **0.981** | 0.084 | 0.575 | 0.687 | 0.669 | 0.326 | 0.498 |
| job_scams | 0.013 | **0.679** | 0.301 | 0.150 | 0.065 | 0.088 | 0.020 |
| phishing | 0.016 | 0.090 | **0.989** | 0.075 | 0.397 | 0.660 | 0.000 |
| political_statements | 0.670 | 0.083 | 0.557 | **0.729** | 0.664 | 0.255 | 0.506 |
| product_reviews | 0.502 | 0.065 | 0.633 | 0.538 | **0.721** | 0.302 | 0.427 |
| sms | 0.592 | 0.080 | 0.577 | 0.322 | 0.241 | **0.988** | 0.504 |
| twitter_rumours | 0.147 | 0.000 | 0.081 | 0.036 | 0.017 | 0.055 | **0.806** |


Trivial all-deceptive F1 per test domain: fake_news 0.603, job_scams 0.081, phishing 0.569, political_statements 0.783, product_reviews 0.667, sms 0.326, twitter_rumours 0.508

No identical text occurs in both a training domain and a different test domain.

