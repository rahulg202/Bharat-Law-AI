# Requirements Document

## Introduction

Astute AI is an AI-powered healthcare platform consisting of two integrated systems: Vaidya AI (a medical search engine for healthcare professionals) and Pharmacov AI (a drug safety monitoring system for pharmaceutical companies). The platform leverages fine-tuned healthcare LLMs to provide evidence-based clinical answers and automate pharmacovigilance workflows.

## Glossary

- **Vaidya_AI**: The medical search engine component that provides evidence-based clinical answers to healthcare professionals
- **Pharmacov_AI**: The drug safety monitoring component that automates pharmacovigilance workflows for pharmaceutical companies
- **ICSR**: Individual Case Safety Report - a document describing a suspected adverse drug reaction
- **MedDRA**: Medical Dictionary for Regulatory Activities - standardized medical terminology
- **Adverse_Event**: An undesirable experience associated with the use of a medical product
- **Signal_Detection**: The process of identifying potential safety issues from aggregated data
- **Evidence_Grader**: Component that assesses and grades the quality of medical evidence
- **Citation_Generator**: Component that creates properly formatted references to source materials
- **Query_Processor**: Component that analyzes and classifies incoming medical queries
- **Safety_Checker**: Component that performs drug interaction and contraindication checks
- **Case_Processor**: Component that handles ICSR intake, extraction, and coding
- **Literature_Monitor**: Component that screens and analyzes medical publications for safety signals
- **Report_Generator**: Component that creates aggregate safety reports and visualizations

## Requirements

### Requirement 1: Medical Query Processing

**User Story:** As a healthcare professional, I want to submit natural language medical queries, so that I can get evidence-based answers without learning complex search syntax.

#### Acceptance Criteria

1. WHEN a healthcare professional submits a medical query, THE Query_Processor SHALL analyze the query using natural language processing
2. WHEN processing a query, THE Query_Processor SHALL identify the medical specialty context
3. WHEN processing a query, THE Query_Processor SHALL detect urgency level of the clinical question
4. WHEN processing a query, THE Query_Processor SHALL classify the query intent (diagnosis, treatment, drug interaction, etc.)
5. IF a query is ambiguous or incomplete, THEN THE Query_Processor SHALL request clarification from the user

### Requirement 2: Multi-Source Medical Data Retrieval

**User Story:** As a healthcare professional, I want the system to search multiple trusted medical sources simultaneously, so that I receive comprehensive and reliable information.

#### Acceptance Criteria

1. WHEN a query is processed, THE Vaidya_AI SHALL search PubMed for relevant medical literature
2. WHEN a query is processed, THE Vaidya_AI SHALL access NIH datasets for clinical data
3. WHEN a query is processed, THE Vaidya_AI SHALL retrieve WHO and CDC guidelines
4. WHEN a query is processed, THE Vaidya_AI SHALL query FDA databases for drug information
5. WHEN retrieving data, THE Vaidya_AI SHALL execute searches in parallel to minimize response time
6. IF a data source is unavailable, THEN THE Vaidya_AI SHALL continue with available sources and indicate which sources were not accessible

### Requirement 3: Evidence Analysis and Grading

**User Story:** As a healthcare professional, I want retrieved evidence to be analyzed and graded for quality, so that I can prioritize the most reliable information.

#### Acceptance Criteria

1. WHEN evidence is retrieved, THE Evidence_Grader SHALL assess study quality using established criteria
2. WHEN grading evidence, THE Evidence_Grader SHALL assign evidence level grades (Level I-V)
3. WHEN multiple studies exist, THE Evidence_Grader SHALL integrate meta-analysis results
4. WHEN grading evidence, THE Evidence_Grader SHALL evaluate statistical significance of findings
5. WHEN grading evidence, THE Evidence_Grader SHALL apply recency weighting to favor newer research
6. WHEN grading evidence, THE Evidence_Grader SHALL verify peer-review status of sources

### Requirement 4: Drug Safety and Interaction Checking

**User Story:** As a healthcare professional, I want automatic safety checks on drug-related queries, so that I can avoid prescribing harmful combinations.

#### Acceptance Criteria

1. WHEN a query involves medications, THE Safety_Checker SHALL detect potential drug-drug interactions
2. WHEN a query involves medications, THE Safety_Checker SHALL generate contraindication alerts
3. WHEN patient context is provided, THE Safety_Checker SHALL cross-reference known allergies
4. WHEN a query involves medications, THE Safety_Checker SHALL check dose adjustment requirements
5. WHEN patient context includes pregnancy or lactation, THE Safety_Checker SHALL flag relevant safety concerns
6. WHEN patient context includes renal or hepatic conditions, THE Safety_Checker SHALL provide appropriate warnings

### Requirement 5: Clinical Guideline Compliance

**User Story:** As a healthcare professional, I want recommendations verified against current clinical guidelines, so that my treatment decisions align with best practices.

#### Acceptance Criteria

1. WHEN generating recommendations, THE Vaidya_AI SHALL verify alignment with ACC/AHA guidelines
2. WHEN generating recommendations, THE Vaidya_AI SHALL check WHO protocol alignment
3. WHEN generating recommendations, THE Vaidya_AI SHALL verify CDC guideline compliance
4. WHEN generating recommendations, THE Vaidya_AI SHALL check specialty-specific standards
5. WHEN guidelines have regional variations, THE Vaidya_AI SHALL indicate applicable regions
6. WHEN guidelines are updated, THE Vaidya_AI SHALL track and display version information

### Requirement 6: Citation Generation

**User Story:** As a healthcare professional, I want all recommendations to include proper citations, so that I can verify sources and document my clinical decisions.

#### Acceptance Criteria

1. WHEN generating an answer, THE Citation_Generator SHALL include PubMed IDs for referenced articles
2. WHEN referencing clinical trials, THE Citation_Generator SHALL include trial registry numbers
3. WHEN generating citations, THE Citation_Generator SHALL include DOI references where available
4. WHEN referencing guidelines, THE Citation_Generator SHALL include guideline versions
5. WHEN generating citations, THE Citation_Generator SHALL display author credentials
6. WHEN generating citations, THE Citation_Generator SHALL show journal impact factors

### Requirement 7: Answer Synthesis and Presentation

**User Story:** As a healthcare professional, I want synthesized answers that integrate multiple sources into actionable recommendations, so that I can make informed clinical decisions quickly.

#### Acceptance Criteria

1. WHEN presenting results, THE Vaidya_AI SHALL integrate information from multiple sources coherently
2. WHEN presenting results, THE Vaidya_AI SHALL provide context-aware responses based on query specialty
3. WHEN presenting results, THE Vaidya_AI SHALL include a plain language summary
4. WHEN presenting treatment options, THE Vaidya_AI SHALL rank options by evidence strength
5. WHEN presenting results, THE Vaidya_AI SHALL provide comparative analysis of treatment approaches
6. WHEN presenting results, THE Vaidya_AI SHALL format output for mobile-responsive delivery

### Requirement 8: Quality Assurance and Confidence Scoring

**User Story:** As a healthcare professional, I want to know the confidence level of AI-generated answers, so that I can appropriately weigh the recommendations.

#### Acceptance Criteria

1. WHEN generating answers, THE Vaidya_AI SHALL perform hallucination detection checks
2. WHEN generating answers, THE Vaidya_AI SHALL provide confidence scores for recommendations
3. WHEN confidence is low, THE Vaidya_AI SHALL flag uncertainty explicitly
4. WHEN sources contain contradictions, THE Vaidya_AI SHALL highlight the contradiction
5. WHEN generating answers, THE Vaidya_AI SHALL validate completeness of the response
6. WHEN confidence is below threshold, THE Vaidya_AI SHALL trigger expert review workflows

### Requirement 9: ICSR Intake and Triage

**User Story:** As a pharmacovigilance specialist, I want automated intake and classification of adverse event reports, so that cases are processed efficiently.

#### Acceptance Criteria

1. WHEN an adverse event report is received, THE Case_Processor SHALL read and parse emails, call transcripts, and PDFs using NLP
2. WHEN processing intake, THE Case_Processor SHALL classify content as AE, PQC, Medical Inquiry, or Non-case
3. WHEN attachments are present, THE Case_Processor SHALL scan and classify documents for ICSR content
4. WHEN ICSR-containing documents are detected, THE Case_Processor SHALL extract text and surface features
5. IF content classification is uncertain, THEN THE Case_Processor SHALL flag for human review

### Requirement 10: ICSR Data Extraction and Coding

**User Story:** As a pharmacovigilance specialist, I want automated extraction and coding of case data, so that I can focus on medical review rather than data entry.

#### Acceptance Criteria

1. WHEN processing an ICSR, THE Case_Processor SHALL extract structured fields using named entity recognition
2. WHEN extracting data, THE Case_Processor SHALL require human approval for every extracted field
3. WHEN coding adverse events, THE Case_Processor SHALL map verbatim terms to MedDRA preferred terms using similarity matching
4. WHEN coding drugs, THE Case_Processor SHALL suggest candidate drug products from WHO drug dictionary
5. WHEN coding is complete, THE Case_Processor SHALL require reviewer selection of final codes
6. WHEN generating narratives, THE Case_Processor SHALL convert structured fields into readable case narratives using LLMs

### Requirement 11: ICSR Quality Control

**User Story:** As a pharmacovigilance specialist, I want automated quality checks on case data, so that submissions are accurate and compliant.

#### Acceptance Criteria

1. WHEN validating a case, THE Case_Processor SHALL check for missing mandatory fields
2. WHEN validating a case, THE Case_Processor SHALL detect mismatched dates
3. WHEN validating a case, THE Case_Processor SHALL identify incorrect gender-pregnancy combinations
4. WHEN validating a case, THE Case_Processor SHALL detect contradictions between narrative and structured fields
5. WHEN duplicates are suspected, THE Case_Processor SHALL compare cases using text similarity and patient characteristics
6. WHEN a potential duplicate is found, THE Case_Processor SHALL produce a similarity score for human decision

### Requirement 12: Literature Monitoring

**User Story:** As a pharmacovigilance specialist, I want automated monitoring of medical literature for safety signals, so that potential issues are identified early.

#### Acceptance Criteria

1. WHEN screening publications, THE Literature_Monitor SHALL classify articles as Relevant or Not Relevant
2. WHEN an article is relevant, THE Literature_Monitor SHALL summarize key events, drug mentions, study type, and patient counts
3. WHEN screening literature, THE Literature_Monitor SHALL detect case-like phrases and drug-event relationships indicating potential ICSRs
4. WHEN a potential ICSR is detected in literature, THE Literature_Monitor SHALL flag for human validation
5. WHEN displaying full text, THE Literature_Monitor SHALL highlight event descriptions, drug exposures, and serious outcomes
6. WHEN prioritizing journals, THE Literature_Monitor SHALL rank by likelihood of producing ICSR cases and covering relevant therapeutic areas

### Requirement 13: Aggregate Reporting and Signal Detection

**User Story:** As a pharmacovigilance specialist, I want automated aggregate analysis and report generation, so that safety trends are identified and regulatory reports are prepared efficiently.

#### Acceptance Criteria

1. WHEN analyzing cases, THE Report_Generator SHALL identify trends in adverse events over time
2. WHEN detecting signals, THE Report_Generator SHALL perform disproportionality analysis and clustering
3. WHEN generating reports, THE Report_Generator SHALL check consistency between narratives, tables, and data sources
4. WHEN generating reports, THE Report_Generator SHALL draft executive summaries, regulatory status sections, and exposure summaries using LLMs
5. WHEN generating visualizations, THE Report_Generator SHALL create time-to-onset plots, cumulative case count graphs, and seriousness distribution charts
6. WHEN drafts are complete, THE Report_Generator SHALL require medical writer review and approval

### Requirement 14: Synergistic Learning Loop

**User Story:** As a platform administrator, I want the two AI systems to improve each other through shared learning, so that accuracy continuously improves.

#### Acceptance Criteria

1. WHEN Vaidya_AI processes medical queries, THE Platform SHALL capture validated medical knowledge for model improvement
2. WHEN Pharmacov_AI processes real-world safety cases, THE Platform SHALL use insights to enhance clinical reasoning in Vaidya_AI
3. WHEN models are fine-tuned, THE Platform SHALL use validated medical literature and regulatory guidelines
4. WHEN updating models, THE Platform SHALL track and version all training data sources
5. WHEN measuring improvement, THE Platform SHALL compare hallucination rates before and after fine-tuning
