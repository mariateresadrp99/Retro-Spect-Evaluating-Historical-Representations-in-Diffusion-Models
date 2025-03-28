## Code for "Retro-Spect: Evaluating Historical Representation in Diffusion Models"

![Evaluation Methodology](./evalutation_methodology.png)

Text-To-Image (TTI) models have become powerful tools for artistic creation and design. However, while existing research predominantly examines their embedded demographic and cultural biases, their ability to accurately represent historical contexts remains largely unexplored. In this work, we introduce a systematic and reproducible methodology for evaluating how TTI systems depict different historical periods across multiple dimensions. Our approach is grounded in HistVis, a curated dataset of 30,000 synthetic images generated by three state-of-the-art diffusion models using carefully designed prompts that depict universal human activities across diverse historical contexts. We evaluate historical depiction along three key dimensions: (1) Implicit Stylistic Associations: examining how models default to certain visual styles for specific periods; 
(2) Historical Consistency: detecting anachronisms such as the depiction of modern objects in historical scenes; and (3) Demographic Representation: comparing generated racial and gender distributions against historically plausible baselines derived from Large Language Models. We find that TTI models frequently stereotype past eras by adding visual stylistic properties not defined in the prompt, while also introducing anachronisms at notable rates and failing to reflect historically plausible demographic patterns. By providing a structured evaluation methodology and empirical insights, this work highlights critical gaps in the historical reasoning of TTI models. We release both the HistVis dataset and the accompanying tools needed to replicate our analysis and support the evaluation of additional TTI systems, laying the foundation for more historically responsible generative models.


## The HistVis Dataset

**HistVis** is a curated dataset of **30,000 synthetic images** generated by state-of-the-art text-to-image (TTI) diffusion models, designed to evaluate how these models represent historical contexts across time. The dataset supports the evaluation methodology introduced in *Retro-Spect: Evaluating Historical Representation in Diffusion Models*.

### Dataset Overview

- **30,000 images** total
- **100 activities** × **10 time periods** × **3 models** × **10 images per pair**
- Prompts follow the format:  
  > *"A person [activity] in the [historical period]"*
- **Time Periods**:  
  - Five centuries (17th–21st)  
  - Five decades (1910s–1990s)
- **Activities**:  
  - 100 universal human activities  
  - Drawn from 20 categories (e.g., art, work, celebration, survival, communication)

Each image is annotated with:
- Prompt metadata
- Activity category
- Time period
- Model identifier

### Dataset Access

