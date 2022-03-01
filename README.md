# NLP PubMed Medical Research Paper Abstract (Randomized Controlled Trial)

A natural language processing model for sequential sentence classification in medical abstracts. 

- The **objective** is to build a deep learning model which makes medical research paper abstract easier to read.

- **Dataset** used in this project is the `PubMed 200k RCT Dataset for Sequential Sentence Classification in Medical Abstract` on arxiv: https://arxiv.org/abs/1710.06071

- The initial deep learning research paper was built with the PubMed 200k RCT.

- **Dataset has about 200,000 labelled Randomized Control Trial abstracts.**

- The **goal of the project** was to build NLP models with the dataset to classify sentences in sequential order.
- As the RCT research papers with unstructured abstracts slows down researchers navigating the literature.
- The unstructured abstracts are sometimes hard to read and understand especially when it can disrupt time management and deadlines.
- This NLP model can classify the abstract sentences into its respective roles:
  
       - Objective
       - Methods 
       - Results
       - Conclusions.
       
       
- The **PubMed 200k RCT Dataset** - https://github.com/Franck-Dernoncourt/pubmed-rct



# Results after NLP processing, sample model prediction from the experiment:

   
**Source** 
      
   **Name**: Randomized Controlled Trial: **RCT of a manualized social treatment for high-functioning autism spectrum disorders**.
             (by Christopher Lopata, Marcus L Thomeer, etc.)
             https://pubmed.ncbi.nlm.nih.gov/20232240/
             
             


**Abstract**: "This RCT examined the efficacy of a manualized social intervention for children with HFASDs. Participants were randomly assigned to treatment or wait-list conditions. Treatment included instruction and therapeutic activities targeting social skills, face-emotion recognition, interest expansion, and interpretation of non-literal language. A response-cost program was applied to reduce problem behaviors and foster skills acquisition. Significant treatment effects were found for five of seven primary outcome measures (parent ratings and direct child measures). Secondary measures based on staff ratings (treatment group only) corroborated gains reported by parents. High levels of parent, child and staff satisfaction were reported, along with high levels of treatment fidelity. Standardized effect size estimates were primarily in the medium and large ranges and favored the treatment group."




# NLP processed abstract after modelling (Model's Predicted Abstract which makes Abstract easier to read)


OBJECTIVE: This RCT examined the efficacy of a manualized social intervention for children with HFASDs.

METHODS: Participants were randomly assigned to treatment or wait-list conditions.

METHODS: Treatment included instruction and therapeutic activities targeting social skills, face-emotion recognition, interest expansion, and interpretation of non-literal language.

METHODS: A response-cost program was applied to reduce problem behaviors and foster skills acquisition.

RESULTS: Significant treatment effects were found for five of seven primary outcome measures (parent ratings and direct child measures).

METHODS: Secondary measures based on staff ratings (treatment group only) corroborated gains reported by parents.

RESULTS: High levels of parent, child and staff satisfaction were reported, along with high levels of treatment fidelity.

RESULTS: Standardized effect size estimates were primarily in the medium and large ranges and favored the treatment group.
