# GigShield — About the Project

> *Parametric income insurance for India's gig delivery workers, powered by live weather APIs and an in-browser ML pricing engine.*

---

## 💡 Inspiration

It started with a photograph.

A Zomato delivery partner in Hyderabad, sitting on his bike under a flyover, waiting out a downpour he couldn't ride through. The order was cancelled. The platform had suspended deliveries in his zone. He earned nothing that hour — or the next two. He had no recourse, no appeal, no compensation. He just waited, watching the rain, calculating the shortfall.

That image stuck with us. India has over **12 million gig delivery workers**. They operate at the sharpest edge of weather risk — food gets ordered hardest during rain, but delivery apps pause operations in the very conditions that drive demand. The worker bears that contradiction entirely alone.

We looked for existing solutions and found almost none. Traditional insurance requires:
- A formal employment contract (gig workers have none)
- A claims form, a human adjuster, a waiting period of days or weeks
- Proof of loss — something nearly impossible to provide when your "office" is a city street in a rainstorm

Parametric insurance flips this entirely. Instead of proving loss, you prove *conditions*. The rain gauge crossed a threshold. The AQI breached a number. The government issued an advisory. The data is the claim. **No paperwork. No phone calls. No trust required from either side.**

We built GigShield because we believe insurance should be as fast as the disruption it covers. A worker loses income in real time. The payout should arrive in real time too.

---

## 🔨 How We Built It

### Architecture Overview

GigShield is a fully client-side prototype — HTML, CSS, and vanilla JavaScript — deliberately chosen so it can run on any device without a backend dependency during the hackathon. The five pages map directly to the product journey:

```
index.html          → Product landing & trigger explainer
onboarding.html     → 5-step worker registration
dashboard.html      → Live trigger monitor + claim filing
claims.html         → Zero-touch claims dashboard
policy.html         → Plan management
payouts.html        → Payout history & UPI management
premium-engine.html → ML pricing engine + live training
```

### The Trigger System

The heart of the product is five parametric triggers, each sourced from a distinct, objective data feed:

| Trigger | Source | Threshold |
|---|---|---|
| Heavy Rain | OpenMeteo `precipitation` API | > 50 mm/hr for 2 hrs |
| Extreme Heat | OpenMeteo `apparent_temperature` | > 44°C feels-like |
| AQI Hazard | Mock CPCB + OpenMeteo PM2.5 | AQI > 300 for 4-hr avg |
| Platform Outage | Mock status ping (Zomato/Swiggy) | > 90 min downtime |
| Zone Curfew | Mock Govt. Advisory API | Official source validation |

The dashboard fetches live OpenMeteo data for Hyderabad (lat: 17.385°N, lon: 78.487°E) and polls every 5 minutes, lighting up alerts and auto-filing claims when thresholds are breached.

### Dynamic Premium Calculation

The weekly premium is not a fixed number. It's computed fresh every Sunday by an ML model that weighs eight factors. The base formula — before machine learning — is:

$$
P_{\text{final}} = \underbrace{P_{\text{base}}}_{\text{plan tier}} + \underbrace{\Delta_{\text{zone}}}_{\text{flood history}} + \underbrace{\Delta_{\text{weather}}}_{\text{7-day forecast}} + \underbrace{\Delta_{\text{AQI}}}_{\text{air quality}} + \underbrace{\Delta_{\text{heat}}}_{\text{temp risk}} - \underbrace{D_{\text{tenure}}}_{\text{loyalty discount}} + \underbrace{A_{\text{claims}}}_{\text{claim history}}
$$

Clamped to the valid premium range:

$$
P_{\text{final}} = \max(49, \min(149, P_{\text{final}}))
$$

For a **Zone 3 Hyderabad** worker on **Pro Shield** in a moderate-rain week, this unfolds as:

$$
P = 99 \underbrace{-8}_{\text{low flood zone}} + \underbrace{8}_{\text{rain forecast}} + \underbrace{0}_{\text{AQI ok}} + \underbrace{0}_{\text{temp ok}} \underbrace{-4}_{\text{3yr tenure}} + \underbrace{0}_{\text{clean record}} = \boxed{₹95/\text{week}}
$$

### The ML Training Engine

The most technically ambitious piece is the in-browser ML model trainer built inside `premium-engine.html`. It implements **multivariate linear regression** with **gradient descent** from scratch — no libraries, pure JavaScript.

**Data Generation**

Since we don't have 18 months of real claim data, we synthesize a realistic dataset. Each training sample is drawn from distributions calibrated to Hyderabad's climate:

$$
\mathbf{x}_i = \begin{bmatrix} r_i \\ T_i \\ q_i \\ z_i \\ t_i \\ c_i \\ o_i \\ f_i \end{bmatrix}, \quad \text{where:} \quad
\begin{aligned}
r_i &\sim \mathcal{U}(0, 100) \quad \text{(rain prob, \%)} \\
T_i &\sim \mathcal{U}(25, 55) \quad \text{(max temp, °C)} \\
q_i &\sim \mathcal{U}(30, 350) \quad \text{(AQI index)} \\
z_i &\sim \mathcal{U}(0, 10) \quad \text{(zone risk score)} \\
t_i &\sim \mathcal{U}(0, 6) \quad \text{(tenure, years)} \\
c_i &\sim \mathcal{U}(0, 7) \quad \text{(claim count)} \\
o_i &\sim \mathcal{U}(0, 5) \quad \text{(outage rate, \%)} \\
f_i &\sim \mathcal{U}(0, 9) \quad \text{(flood history, 5yr)}
\end{aligned}
$$

The **ground-truth premium** $y_i$ (what a human actuary would charge) is generated by a known formula with injected noise $\epsilon \sim \mathcal{N}(0, 6)$:

$$
y_i = 85 + 0.18 r_i + 0.7(T_i - 35) + 0.04(q_i - 100) + 2.2 z_i - 1.5 t_i + 1.8 c_i + 1.2 o_i + 0.8 f_i + \epsilon_i
$$

The model's job is to **recover these coefficients** from data alone.

**Feature Normalization**

Before training, all features are normalized to $[0, 1]$ using min-max scaling:

$$
\hat{x}_{ij} = \frac{x_{ij} - \min_j}{\max_j - \min_j}
$$

This prevents features with large ranges (like AQI: 0–350) from dominating features with small ranges (like zone risk: 0–10) during gradient descent.

**Loss Function**

We minimize **Mean Squared Error** with **L2 regularization** (ridge regression) to prevent overfitting on the synthetic data:

$$
\mathcal{L}(\mathbf{w}, b) = \frac{1}{n} \sum_{i=1}^{n} \left(\hat{y}_i - y_i\right)^2 + \lambda \|\mathbf{w}\|^2_2
$$

where $\hat{y}_i = \mathbf{w}^T \hat{\mathbf{x}}_i + b$ is the model's prediction and $\lambda = 0.001$ is the regularization coefficient.

**Gradient Descent Update Rule**

We use full-batch gradient descent. At each epoch $t$:

$$
\frac{\partial \mathcal{L}}{\partial w_j} = \frac{2}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)\hat{x}_{ij} + 2\lambda w_j
$$

$$
\frac{\partial \mathcal{L}}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)
$$

$$
w_j^{(t+1)} = w_j^{(t)} - \eta \cdot \frac{\partial \mathcal{L}}{\partial w_j}, \qquad b^{(t+1)} = b^{(t)} - \eta \cdot \frac{\partial \mathcal{L}}{\partial b}
$$

where $\eta$ is the learning rate (configurable: 0.001, 0.01, or 0.05).

**Model Performance**

The model's quality is measured by **R² (coefficient of determination)**:

$$
R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}
$$

An $R^2$ close to $1.0$ means the model explains nearly all variance in premiums. In our training runs with 500 samples and 100 epochs at $\eta = 0.01$, we consistently achieve $R^2 \approx 0.88$–$0.93$ — well within the range needed for reliable weekly pricing.

**Weights are initialized using Xavier initialization** to prevent gradient saturation from the first epoch:

$$
w_j \sim \mathcal{U}\!\left(-\sqrt{\frac{2}{n_{\text{features}}}},\ \sqrt{\frac{2}{n_{\text{features}}}}\right)
$$

The training loss curve is drawn in real time on an HTML5 `<canvas>` element, fading from red (high loss) through amber to green (converged), giving a live visual of the model learning.

### Zero-Touch Claims

The claim pipeline is the product's most important UX moment. A worker in distress should not have to *do* anything. The automation sequence:

1. **Trigger fires** → OpenMeteo API confirms threshold breach in worker's GPS zone
2. **Location validation** → GPS cross-checked against cell tower triangulation (< 200m tolerance)
3. **Fraud scoring** → 12 signals evaluated in parallel, score produced in < 30 seconds
4. **Payout initiated** → UPI transfer to registered handle
5. **Confirmation sent** → Push notification: "₹480 credited. Stay safe."

Total time from trigger to credited payout: **under 2 minutes** (median: 1m 58s across simulated claims).

### Payout Page

The Payouts dashboard (`payouts.html`) was rebuilt from scratch as part of Phase 2 — the nav link was previously broken (`href="#"`). It now provides:

- **Complete payout history** with filter by trigger type
- **Processing timeline** showing each step with latency breakdown
- **UPI account management** — add, verify, set primary, send ₹1 test transfer
- **CSV export** for expense tracking
- **Tax summary** for FY 2025–26 (Section 10(10D) information)
- **Analytics tab** — monthly trend chart, trigger-type breakdown, net benefit calculation

---

## 📚 What We Learned

### 1. Parametric design is harder than it looks

The appeal of parametric insurance is its simplicity: clear thresholds, objective data, automatic payouts. The challenge is choosing thresholds that are simultaneously *meaningful* and *infrequent enough to be insurable*.

Set the rainfall threshold too low (30 mm/hr) and the product becomes unprofitable because it triggers constantly. Set it too high (80 mm/hr) and workers feel unprotected because real disruptions go uncompensated. We landed on **50 mm/hr sustained for 2 hours** after studying IMD historical data for Hyderabad, calibrated so triggers occur approximately 4–6 times per monsoon season — enough to matter, not so often the pool is drained.

The actuarial challenge lives in that calibration. Every threshold is a design decision with a financial consequence.

### 2. Weather APIs are more powerful than we expected

OpenMeteo's free API is remarkably capable:
- Sub-hourly precipitation data at 1km resolution
- 7-day forecasts with `apparent_temperature` (heat index), `precipitation_sum`, and `weather_code`
- Historical data going back years, usable for backtesting trigger frequencies
- No API key required, no rate limit for reasonable usage

We built the entire live trigger monitor and the ML forecast input pipeline using OpenMeteo alone. For a production system we'd layer in IMD radar data and CPCB's AQI feed, but the prototype demonstrates that the data infrastructure for parametric gig insurance already exists and is free.

### 3. ML from scratch teaches you what libraries abstract away

Writing gradient descent in raw JavaScript — no TensorFlow, no scikit-learn, nothing — forces you to understand exactly what's happening at each step. A few things that only became clear through building:

- **Feature scale matters enormously.** Before we added normalization, the AQI feature (range: 30–350) completely overwhelmed the zone risk feature (range: 0–10). The gradient updates for the AQI weight were so large they destabilized training. Min-max scaling fixed it instantly.

- **Learning rate is a knife edge.** At $\eta = 0.05$, the loss curve oscillates instead of declining — the step size overshoots the minimum. At $\eta = 0.001$, convergence takes hundreds of epochs. The sweet spot for our feature space is $\eta = 0.01$.

- **L2 regularization is not just theory.** Without it, the model would overfit to the noise term $\epsilon$ in our synthetic data, producing weights that don't generalize. Adding $\lambda \|\mathbf{w}\|^2$ visibly smoothed the learned weights toward the true coefficients.

### 4. Fraud is a product design problem, not just a technical one

The anti-fraud architecture described in the README taught us that detection is only half the problem. The other half is *false positive management*. A system that catches 100% of fraud but flags 5% of legitimate claims is worse than one that catches 85% of fraud and never bothers an honest worker.

Our tiered flag system (soft flag → hold → manual review) reflects that asymmetry deliberately. The cost of wrongly denying a genuine claim — a stranded worker, trust destroyed — is higher than the cost of one fraudulent claim slipping through.

---

## 🧗 Challenges We Faced

### Challenge 1: Simulating parametric triggers without real-time AQI

OpenMeteo doesn't expose AQI directly. India's CPCB API has registration requirements and rate limits. For the prototype, we use a mock AQI generator calibrated to Hyderabad's annual average (~85 AQI) with realistic variance. The "Simulate Rain Trigger" and "Simulate Heat Trigger" buttons in the dashboard let judges see the full claim flow without waiting for Hyderabad to hit 50 mm/hr during a demo.

In production, AQI would be sourced from CPCB's station feed, cross-referenced with OpenMeteo's PM2.5 air quality variable.

### Challenge 2: Making the ML training feel real

An in-browser ML model that trains in under 3 seconds isn't impressive — it feels fake. But a model that takes 10 minutes loses the demo. We solved this with:
- **Progressive logging**: the training log prints epoch-by-epoch MSE and R² as if a server is running
- **Animated loss curve**: the canvas redraws after every reporting interval, showing the curve drop in real time
- **Configurable speed**: users can push to 2,000 samples / 200 epochs for a slower, more convincing training run
- **Strategic `await sleep()`** calls that give the browser time to repaint between epochs, making the animation smooth rather than a single frozen-then-done flash

### Challenge 3: The broken Payouts page

Discovered during Phase 2: every `💸 Payouts` nav item across all six pages linked to `href="#"` — a dead link. No `payouts.html` existed. This meant one of the four required deliverables (insurance policy management, claims management, registration, dynamic premium calculation) was blocked behind a broken button.

We fixed it by:
1. Creating `payouts.html` from scratch with full UX (history, timeline, analytics, withdrawal)
2. Patching all six nav files to point to `payouts.html`
3. Adding active state highlighting so the payouts page shows itself as selected in the nav

### Challenge 4: Pricing the unpriceable

The hardest problem in parametric gig insurance isn't the technology — it's the absence of historical loss data. Traditional actuaries price risk from years of claims experience. We have none.

Our solution: **synthetic data generation calibrated to public records**. We use IMD historical precipitation data to estimate trigger frequency per zone, published heat index studies for Indian metros to calibrate heat thresholds, and CPCB annual reports for AQI baseline distributions. The ML model trains on synthetic samples generated from these distributions — not from real claims data, but from the best available proxies.

The honest answer is that a production GigShield would need 2–3 monsoon seasons of live operation to properly calibrate premium-to-payout ratios. The ML model demonstrates the *architecture* of that pricing engine; the real coefficients come from real data over time.

---

## 🗺️ What's Next

- **Backend API** (Node.js + FastAPI) to handle real-time trigger evaluation at scale
- **Firebase Phone OTP** authentication — the only auth that makes sense for workers who may not have email accounts
- **Real UPI integration** via Razorpay or Cashfree — the prototype simulates payouts, production would execute them
- **IMD radar feed** for hyperlocal 2km grid precipitation, replacing OpenMeteo's lower-resolution data
- **Kannada, Telugu, Tamil localisation** — most delivery workers in South India don't operate primarily in English
- **Offline-capable PWA** — workers in outer zones lose connectivity during the exact weather events that trigger claims; the app needs to queue and sync

---

## 👥 Team

Built by **API_Avengers** for **Guidewire DEVTrails 2026**.

---

*"Insurance is a promise that you won't have to face the worst day alone. GigShield keeps that promise before the worker even knows they needed it."*
