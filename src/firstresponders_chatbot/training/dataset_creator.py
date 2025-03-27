"""
Dataset creator module for the FirstRespondersChatbot.

This module contains the DatasetCreator class for generating a dataset
for fine-tuning Llama 3 models.
"""

import json
import logging
import os
import random
import re
import nltk
from pathlib import Path
from typing import Dict, List, Tuple, Any
from nltk.corpus import stopwords
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetCreator:
    """Dataset creator for the FirstRespondersChatbot project."""

    def __init__(
        self,
        input_file: str = "data/preprocessed_data.json",
        output_file: str = "data/pseudo_data.json",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        min_content_length: int = 100,  # Minimum content length to consider
        similarity_threshold: float = 0.65,  # Optimized threshold
        max_examples_per_doc: int = 5,  # Generate examples per document
        model_format: str = "llama3",  # Format always set to llama3
    ):
        """
        Initialize the dataset creator.

        Args:
            input_file: Path to the preprocessed data file
            output_file: Path to save the generated dataset
            train_ratio: Ratio of data to use for training
            val_ratio: Ratio of data to use for validation
            test_ratio: Ratio of data to use for testing
            min_content_length: Minimum content length to consider for a document
            similarity_threshold: Threshold for detecting similar documents
            max_examples_per_doc: Maximum examples to generate from a document
            model_format: Format to use for the dataset (always llama3)
        """
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.min_content_length = min_content_length
        self.similarity_threshold = similarity_threshold
        self.max_examples_per_doc = max_examples_per_doc
        self.model_format = model_format

        # Download NLTK resources if needed
        try:
            nltk.data.find("tokenizers/punkt")
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("punkt")
            nltk.download("stopwords")

        # First responder domain-specific categories for better organization
        self.domain_categories = {
            "emergency_medical": [
                "CPR",
                "AED",
                "first aid",
                "paramedic",
                "EMT",
                "trauma",
                "medical emergency",
                "triage",
                "patient assessment",
                "vital signs",
            ],
            "fire_safety": [
                "firefighter",
                "fire safety",
                "fire prevention",
                "SCBA",
                "fire suppression",
                "fire hazard",
                "building evacuation",
                "firefighting",
            ],
            "incident_command": [
                "incident command",
                "ICS",
                "emergency management",
                "command structure",
                "emergency operations",
                "EOC",
                "incident commander",
            ],
            "hazardous_materials": [
                "hazmat",
                "hazardous materials",
                "chemical exposure",
                "decontamination",
                "biological hazard",
                "radiological hazard",
                "hazard assessment",
            ],
            "equipment_ppe": [
                "PPE",
                "protective equipment",
                "respirator",
                "safety gear",
                "helmet",
                "bunker gear",
                "turnout gear",
                "safety equipment",
            ],
            "disaster_response": [
                "natural disaster",
                "disaster response",
                "mass casualty",
                "evacuation",
                "search and rescue",
                "emergency shelter",
                "disaster recovery",
            ],
            "protocols_procedures": [
                "protocol",
                "procedure",
                "standard operating procedure",
                "SOP",
                "emergency procedure",
                "guideline",
                "policy",
                "safety procedure",
            ],
        }

        # Ensure ratios sum to 1
        total_ratio = train_ratio + val_ratio + test_ratio
        if not 0.999 <= total_ratio <= 1.001:  # Allow for floating point errors
            raise ValueError(f"Ratios must sum to 1, got {total_ratio}")

    def _extract_keywords(self, text, top_n=10):
        """
        Extract important keywords from text.

        Args:
            text: Text to extract keywords from
            top_n: Number of top keywords to extract

        Returns:
            List of keywords
        """
        # Tokenize and lowercase
        tokens = nltk.word_tokenize(text.lower())

        # Remove stopwords and short words
        stop_words = set(stopwords.words("english"))
        tokens = [
            t for t in tokens if t.isalpha() and t not in stop_words and len(t) > 3
        ]

        # Count word frequencies
        word_freq = Counter(tokens)

        # Get top keywords
        keywords = [word for word, freq in word_freq.most_common(top_n)]

        return keywords

    def _clean_text(self, text):
        """
        Clean and normalize text.

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Fix common OCR errors
        text = text.replace("|", "I").replace("l", "I")

        # Remove page numbers, headers, etc.
        text = re.sub(r"\b[Pp]age\s+\d+\b", "", text)
        text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)

        return text

    def _calculate_text_similarity(self, text1, text2):
        """
        Calculate similarity between two texts using keyword overlap.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        # Extract keywords
        keywords1 = set(self._extract_keywords(text1))
        keywords2 = set(self._extract_keywords(text2))

        # Calculate Jaccard similarity
        if not keywords1 or not keywords2:
            return 0.0

        intersection = keywords1.intersection(keywords2)
        union = keywords1.union(keywords2)

        return len(intersection) / len(union)

    def _filter_documents(self, documents):
        """
        Filter and deduplicate documents.

        Args:
            documents: List of document dictionaries

        Returns:
            Filtered list of documents
        """
        logger.info(f"Starting document filtering with {len(documents)} documents")

        # Filter by length
        filtered_docs = [
            doc
            for doc in documents
            if "content" in doc
            and len(doc["content"].strip()) >= self.min_content_length
        ]

        logger.info(f"After length filtering: {len(filtered_docs)} documents")

        # Clean text
        logger.info("Cleaning document text...")
        cleaned_count = 0
        for doc in filtered_docs:
            doc["content"] = self._clean_text(doc["content"])
            cleaned_count += 1
            if cleaned_count % 1000 == 0:
                logger.info(f"Cleaned {cleaned_count}/{len(filtered_docs)} documents")

        logger.info(f"Completed text cleaning for {len(filtered_docs)} documents")

        # Deduplicate similar documents
        logger.info("Starting document deduplication...")
        unique_docs = []
        duplicate_count = 0
        processed_count = 0

        for doc in filtered_docs:
            processed_count += 1
            if processed_count % 500 == 0:
                logger.info(
                    f"Deduplication progress: {processed_count}/{len(filtered_docs)} documents (found {duplicate_count} duplicates)"
                )

            is_duplicate = False
            for unique_doc in unique_docs:
                similarity = self._calculate_text_similarity(
                    doc["content"], unique_doc["content"]
                )
                if similarity > self.similarity_threshold:
                    is_duplicate = True
                    duplicate_count += 1
                    break

            if not is_duplicate:
                unique_docs.append(doc)

        logger.info(f"Deduplication complete: found {duplicate_count} duplicates")
        logger.info(f"After deduplication: {len(unique_docs)} documents remain")
        return unique_docs

    def generate_question_answer_pairs(
        self, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Generate question-answer pairs from documents.

        Args:
            documents: List of document dictionaries

        Returns:
            List of question-answer pairs
        """
        logger.info("Starting question-answer pair generation")
        qa_pairs = []

        # Filter and clean documents
        filtered_documents = self._filter_documents(documents)

        # Question templates enhanced for first responder domain knowledge
        question_templates = {
            "factual": [
                "What information can you provide about {}?",
                "What are the key points about {}?",
                "What does the manual say about {}?",
                "What is the primary purpose of {}?",
                "What is the core function of {} in emergency response?",
                "What are the critical components of {}?",
                "How is {} defined in first responder contexts?",
            ],
            "procedural": [
                "What procedures are involved with {}?",
                "How should first responders handle situations involving {}?",
                "What are the steps for {}?",
                "What is the proper protocol for {}?",
                "What is the standard operating procedure for managing {}?",
                "How do first responders safely approach a situation with {}?",
                "What sequence of actions should be taken for {}?",
            ],
            "explanatory": [
                "Can you explain the concept of {} in simple terms?",
                "How would you describe {} to a new first responder?",
                "Explain {} as if I'm a trainee first responder.",
                "What specific hazards does {} protect against?",
                "How does {} minimize risk of injury or fatality?",
                "What makes {} important in emergency situations?",
                "Why do first responders need to understand {}?",
            ],
            "application": [
                "How is {} applied in emergency situations?",
                "What are the best practices for {} in emergency situations?",
                "As a first responder, what should I know about {}?",
                "How would I implement {} during an incident?",
                "When would I use {} in the field?",
                "What considerations are important when using {} in an emergency?",
                "How do I properly maintain and use {}?",
            ],
            "scenario_based": [
                "In a mass casualty incident, how would {} be utilized?",
                "During a structural fire, how should {} be approached?",
                "In a hazardous materials incident, what role does {} play?",
                "During patient extrication, how is {} implemented?",
                "In a multi-agency response, how is {} coordinated?",
                "During search and rescue operations, how is {} used effectively?",
            ],
            "safety_focused": [
                "What safety precautions should be taken when using {}?",
                "How does {} help prevent injuries to first responders?",
                "What are the safety limitations of {}?",
                "What risks are associated with improper use of {}?",
                "How can first responders safely implement {}?",
            ],
        }

        # Optimized instruction templates for Llama 2
        instruction_templates = [
            "You are a first responders chatbot. Answer the following question based on the provided context.",
            "As a first responder training assistant, use only the context below to answer the question accurately.",
            "You are a specialized AI assistant for first responders. Answer this question using only the information in the following context.",
            "Based solely on the provided documents, answer this question about first responder procedures.",
            "You are an emergency services training assistant. Using only the following reference materials, answer this question.",
        ]

        # Categorize documents by domain for better context grouping
        categorized_docs = self._categorize_documents(filtered_documents)

        # Group similar documents to create synthetic context
        logger.info("Grouping similar documents by category for better context")
        grouped_docs = []

        # Create groups within each category
        for category, docs in categorized_docs.items():
            category_groups = self._group_similar_documents(
                docs, group_size=3, num_groups=100
            )
            grouped_docs.extend(category_groups)
            logger.info(
                f"Created {len(category_groups)} document groups for category: {category}"
            )

        # Add remaining document groups using standard grouping
        if len(grouped_docs) < 500:
            additional_groups = self._group_similar_documents(
                filtered_documents, group_size=3, num_groups=500 - len(grouped_docs)
            )
            grouped_docs.extend(additional_groups)

        logger.info(
            f"Created {len(grouped_docs)} total document groups for context generation"
        )

        # Track generated topics to avoid duplicates
        generated_topics = set()

        # Process each document group
        logger.info("Generating question-answer pairs from document groups")
        doc_group_count = 0

        for doc_group in grouped_docs:
            doc_group_count += 1

            # Log progress every 50 document groups
            if doc_group_count % 50 == 0:
                logger.info(
                    f"Processed {doc_group_count}/{len(grouped_docs)} document groups, generated {len(qa_pairs)} QA pairs so far"
                )

            if len(doc_group) < 1:
                continue

            # Use the first document to extract keywords
            primary_content = doc_group[0]["content"]
            if (
                not primary_content
                or len(primary_content.strip()) < self.min_content_length
            ):
                continue

            # Get file name for domain-specific question generation
            file_name = (
                doc_group[0]["meta"].get("file_name", "").lower()
                if "meta" in doc_group[0]
                else ""
            )

            # Extract potential topics from keywords with domain focus
            keywords = self._extract_keywords(
                primary_content, top_n=7
            )  # Increased from 5 to 7

            # Add document title-based keywords if available
            if file_name:
                # Extract key terms from filename
                file_terms = re.findall(r"[a-zA-Z]{3,}", file_name)
                for term in file_terms:
                    if term.lower() not in ["pdf", "doc", "txt", "the", "and", "for"]:
                        keywords.append(term)

            # Use domain categories to identify relevant domain terms in content
            for category, terms in self.domain_categories.items():
                for term in terms:
                    if term.lower() in primary_content.lower() and term not in keywords:
                        keywords.append(term)

            # Generate examples based on each keyword topic
            for keyword in keywords:
                if keyword in generated_topics:
                    continue

                # Add to tracking set
                generated_topics.add(keyword)

                # Create context similar to inference time
                context_text = ""
                # Use all docs in the group for richer context
                for i, doc in enumerate(doc_group):
                    meta_info = ""
                    if "meta" in doc and doc["meta"]:
                        if "file_name" in doc["meta"]:
                            meta_info += f" [Source: {doc['meta']['file_name']}]"
                        if "page_number" in doc["meta"]:
                            meta_info += f" [Page: {doc['meta']['page_number']}]"

                    context_text += (
                        f"\n### Document {i+1}{meta_info}:\n{doc['content']}\n"
                    )

                # Identify relevant question types based on keyword and content
                relevant_q_types = self._identify_relevant_question_types(
                    keyword, primary_content
                )

                # If no specific types identified, use all types
                if not relevant_q_types:
                    relevant_q_types = list(question_templates.keys())

                # Generate more questions for important keywords
                num_questions = (
                    min(3, len(relevant_q_types)) if len(primary_content) > 500 else 1
                )

                # Generate questions for different question types
                for q_type in random.sample(relevant_q_types, num_questions):
                    templates = question_templates[q_type]

                    # Generate the question
                    q_template = random.choice(templates)
                    question = q_template.format(keyword)

                    # Choose a random instruction template
                    instruction = random.choice(instruction_templates)

                    # Create the full prompt - optimized for Llama 2 format
                    prompt = f"""{instruction}

Context:
{context_text}

Question: {question}"""

                    # Generate more structured answers for better learning
                    answer = self._generate_structured_answer(
                        keyword, doc_group, q_type
                    )

                    qa_pair = {
                        "question": prompt,
                        "answer": answer,
                    }
                    qa_pairs.append(qa_pair)

        logger.info(f"Generated {len(qa_pairs)} question-answer pairs")

        # Final quality check
        filtered_pairs = self._filter_qa_pairs(qa_pairs)
        logger.info(
            f"After quality filtering: {len(filtered_pairs)} question-answer pairs"
        )

        return filtered_pairs

    def _categorize_documents(
        self, documents: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Categorize documents by domain category.

        Args:
            documents: List of document dictionaries

        Returns:
            Dictionary mapping categories to document lists
        """
        categorized = {category: [] for category in self.domain_categories.keys()}
        categorized["general"] = []  # Default category

        for doc in documents:
            content = doc.get("content", "").lower()
            assigned = False

            # Assign to categories based on keyword matches
            for category, terms in self.domain_categories.items():
                for term in terms:
                    if term.lower() in content:
                        categorized[category].append(doc)
                        assigned = True
                        break
                if assigned:
                    break

            # If no category matched, add to general
            if not assigned:
                categorized["general"].append(doc)

        # Log category counts
        for category, docs in categorized.items():
            logger.info(f"Category '{category}': {len(docs)} documents")

        return categorized

    def _identify_relevant_question_types(
        self, keyword: str, content: str
    ) -> List[str]:
        """
        Identify which question types are most relevant for a given keyword and content.

        Args:
            keyword: The keyword/topic
            content: The document content

        Returns:
            List of relevant question types
        """
        content_lower = content.lower()
        keyword_lower = keyword.lower()
        relevant_types = []

        # Check for procedural content
        if any(
            term in content_lower
            for term in [
                "step",
                "procedure",
                "protocol",
                "guideline",
                "process",
                "sequence",
            ]
        ):
            relevant_types.append("procedural")

        # Check for application examples
        if any(
            term in content_lower
            for term in ["use", "apply", "implement", "deploy", "utilize"]
        ):
            relevant_types.append("application")

        # Check for safety information
        if any(
            term in content_lower
            for term in ["safety", "hazard", "risk", "danger", "protect", "prevention"]
        ):
            relevant_types.append("safety_focused")

        # Check for scenario descriptions
        if any(
            term in content_lower
            for term in [
                "scenario",
                "situation",
                "incident",
                "event",
                "emergency",
                "response",
            ]
        ):
            relevant_types.append("scenario_based")

        # Always include factual for basic information
        relevant_types.append("factual")

        # Check for explanatory needs
        if len(keyword.split()) > 1 or len(keyword) > 10:
            relevant_types.append("explanatory")

        # Ensure we have at least 2 question types
        if len(relevant_types) < 2:
            if "explanatory" not in relevant_types:
                relevant_types.append("explanatory")

        return relevant_types

    def _generate_structured_answer(
        self, keyword: str, doc_group: List[Dict[str, Any]], question_type: str
    ) -> str:
        """
        Generate a structured answer based on keyword, documents and question type.

        Args:
            keyword: The keyword/topic
            doc_group: The group of documents for this question
            question_type: The type of question being asked

        Returns:
            Structured answer text
        """
        # Extract relevant content from all documents
        combined_content = ""
        for doc in doc_group:
            combined_content += " " + doc.get("content", "")

        # Get sentences that mention the keyword
        sentences = nltk.sent_tokenize(combined_content)
        relevant_sentences = [s for s in sentences if keyword.lower() in s.lower()]

        # If no direct keyword matches, use sentences that might be topically related
        if not relevant_sentences:
            # Look for related terms
            related_terms = []
            for category, terms in self.domain_categories.items():
                for term in terms:
                    if (
                        term.lower() in keyword.lower()
                        or keyword.lower() in term.lower()
                    ):
                        related_terms.extend(terms)

            # Get sentences with related terms
            if related_terms:
                relevant_sentences = [
                    s
                    for s in sentences
                    if any(term.lower() in s.lower() for term in related_terms)
                ]

        # If still no relevant sentences, use the first document's content
        if not relevant_sentences and doc_group:
            primary_content = doc_group[0].get("content", "")
            # Take first few sentences as summary
            relevant_sentences = nltk.sent_tokenize(primary_content)[:3]

        # Craft different answer formats based on question type
        if question_type == "factual":
            return self._format_factual_answer(keyword, relevant_sentences)
        elif question_type == "procedural":
            return self._format_procedural_answer(keyword, relevant_sentences)
        elif question_type == "explanatory":
            return self._format_explanatory_answer(keyword, relevant_sentences)
        elif question_type == "application":
            return self._format_application_answer(keyword, relevant_sentences)
        elif question_type == "scenario_based":
            return self._format_scenario_answer(keyword, relevant_sentences)
        elif question_type == "safety_focused":
            return self._format_safety_answer(keyword, relevant_sentences)
        else:
            # Default format
            return self._format_factual_answer(keyword, relevant_sentences)

    def _format_factual_answer(self, keyword: str, sentences: List[str]) -> str:
        """Format a factual answer with key information."""
        if not sentences:
            return f"Based on the provided information, {keyword} is an important concept in emergency response, but specific details are not available in the current context."

        introduction = f"Based on the provided information, {keyword} refers to "
        main_content = " ".join(sentences[:3])  # Use up to 3 sentences

        # Ensure the answer is complete
        if not main_content.strip().endswith((".", "!", "?")):
            main_content = main_content.strip() + "."

        return f"{introduction}{main_content}"

    def _format_procedural_answer(self, keyword: str, sentences: List[str]) -> str:
        """Format a procedural answer with steps."""
        if not sentences:
            return f"The procedure for {keyword} requires following proper protocols as outlined in first responder training, though specific steps are not detailed in the provided context."

        introduction = (
            f"The procedure for handling {keyword} involves the following steps:\n\n"
        )

        # Try to identify and format steps
        steps = []
        for i, sentence in enumerate(sentences[:5]):  # Use up to 5 sentences
            steps.append(f"{i+1}. {sentence.strip()}")

        main_content = "\n".join(steps)

        return f"{introduction}{main_content}"

    def _format_explanatory_answer(self, keyword: str, sentences: List[str]) -> str:
        """Format an explanatory answer that breaks down the concept."""
        if not sentences:
            return f"{keyword} is a concept used in emergency response scenarios. While the exact definition is not provided in the current context, it is generally related to first responder operations and protocols."

        introduction = f"{keyword} can be explained as follows:\n\n"

        # Combine sentences into a cohesive explanation
        explanation = " ".join(sentences[:4])  # Use up to 4 sentences

        # Add a conclusion if we have enough content
        if len(sentences) > 2:
            conclusion = "\n\nUnderstanding this concept is essential for effective emergency response operations."
        else:
            conclusion = ""

        return f"{introduction}{explanation}{conclusion}"

    def _format_application_answer(self, keyword: str, sentences: List[str]) -> str:
        """Format an answer about how something is applied in practice."""
        if not sentences:
            return f"In practical emergency situations, {keyword} would be applied according to protocols and training, though specifics are not detailed in the provided materials."

        introduction = f"In emergency situations, {keyword} is applied as follows:\n\n"

        # Format application points
        applications = []
        for i, sentence in enumerate(sentences[:4]):  # Use up to 4 sentences
            applications.append(f"• {sentence.strip()}")

        main_content = "\n".join(applications)

        return f"{introduction}{main_content}"

    def _format_scenario_answer(self, keyword: str, sentences: List[str]) -> str:
        """Format an answer in the context of an emergency scenario."""
        if not sentences:
            return f"In an emergency scenario involving {keyword}, first responders would follow established protocols and utilize their training to ensure safety and effective response."

        introduction = f"In a scenario involving {keyword}:\n\n"

        # Create a mini-scenario from sentences
        scenario = " ".join(sentences[:5])  # Use up to 5 sentences

        # Add a practical conclusion
        conclusion = f"\n\nFirst responders would need to maintain situational awareness and follow protocols when dealing with {keyword} in this type of scenario."

        return f"{introduction}{scenario}{conclusion}"

    def _format_safety_answer(self, keyword: str, sentences: List[str]) -> str:
        """Format an answer focusing on safety aspects."""
        if not sentences:
            return f"Safety considerations for {keyword} are paramount in emergency response. While specific details aren't provided in the current context, standard safety protocols would apply."

        introduction = (
            f"For {keyword}, the following safety considerations are important:\n\n"
        )

        # Format safety points
        safety_points = []
        for i, sentence in enumerate(sentences[:4]):  # Use up to 4 sentences
            safety_points.append(f"• {sentence.strip()}")

        main_content = "\n".join(safety_points)

        # Add safety emphasis
        conclusion = "\n\nAlways prioritize personal safety and follow established protocols when dealing with emergency situations."

        return f"{introduction}{main_content}{conclusion}"

    def _filter_qa_pairs(self, qa_pairs):
        """
        Filter QA pairs for quality.

        Args:
            qa_pairs: List of QA pairs

        Returns:
            Filtered list of QA pairs
        """
        logger.info(f"Starting QA pair quality filtering on {len(qa_pairs)} pairs")
        filtered = []
        rejected_short_answer = 0
        rejected_question_in_answer = 0

        for pair in qa_pairs:
            # Ensure minimum length for answers
            if len(pair["answer"]) < 50:
                rejected_short_answer += 1
                continue

            # Skip if question appears in answer verbatim
            if pair["question"] in pair["answer"]:
                rejected_question_in_answer += 1
                continue

            # Add to filtered list
            filtered.append(pair)

        logger.info(
            f"QA filtering stats: {rejected_short_answer} pairs rejected for short answers"
        )
        logger.info(
            f"QA filtering stats: {rejected_question_in_answer} pairs rejected for question in answer"
        )
        logger.info(
            f"After quality filtering: {len(filtered)} question-answer pairs kept"
        )

        return filtered

    def _group_similar_documents(
        self,
        documents: List[Dict[str, Any]],
        group_size: int = 3,
        num_groups: int = 500,
    ) -> List[List[Dict[str, Any]]]:
        """
        Group similar documents together for context creation.
        Uses topic similarity for better grouping than random.

        Args:
            documents: List of document dictionaries
            group_size: Number of documents per group
            num_groups: Maximum number of groups to create

        Returns:
            List of document groups
        """
        logger.info(f"Starting document grouping with {len(documents)} documents")

        # Extract keywords for each document
        doc_keywords = []
        for i, doc in enumerate(documents):
            if i % 1000 == 0 and i > 0:
                logger.info(f"Extracted keywords for {i}/{len(documents)} documents")

            keywords = self._extract_keywords(doc["content"], top_n=5)
            doc_keywords.append((doc, set(keywords)))

        logger.info(f"Completed keyword extraction for {len(documents)} documents")

        # Create groups
        groups = []
        remaining_docs = doc_keywords.copy()

        group_count = 0
        logger.info(f"Creating document groups (target: up to {num_groups} groups)")

        while len(remaining_docs) >= 1 and len(groups) < num_groups:
            # Log progress every 50 groups
            if len(groups) % 50 == 0 and len(groups) > 0:
                logger.info(f"Created {len(groups)} document groups so far")

            # Start a new group with the first document
            current_group = [remaining_docs[0][0]]
            current_keywords = remaining_docs[0][1]
            del remaining_docs[0]

            # Find similar documents
            for _ in range(min(group_size - 1, len(remaining_docs))):
                best_idx = -1
                best_similarity = -1

                for i, (doc, keywords) in enumerate(remaining_docs):
                    # Calculate keyword overlap
                    if not current_keywords or not keywords:
                        similarity = 0
                    else:
                        intersection = current_keywords.intersection(keywords)
                        union = current_keywords.union(keywords)
                        similarity = len(intersection) / len(union)

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_idx = i

                if best_idx >= 0:
                    # Add document to group
                    doc, keywords = remaining_docs[best_idx]
                    current_group.append(doc)
                    current_keywords = current_keywords.union(keywords)
                    del remaining_docs[best_idx]
                else:
                    break

            if len(current_group) > 0:
                groups.append(current_group)

        # Add any remaining documents as single-doc groups
        for doc, _ in remaining_docs:
            groups.append([doc])

        return groups

    def load_preprocessed_data(self) -> List[Dict[str, Any]]:
        """
        Load preprocessed data from JSON file.

        Returns:
            List of document dictionaries
        """
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")

        with open(self.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"Loaded {len(data)} documents from {self.input_file}")
        return data

    def split_data(
        self, qa_pairs: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Split data into training, validation, and test sets.

        Args:
            qa_pairs: List of question-answer pairs

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        # Shuffle data
        random.shuffle(qa_pairs)

        # Calculate split indices
        n_samples = len(qa_pairs)
        train_idx = int(n_samples * self.train_ratio)
        val_idx = train_idx + int(n_samples * self.val_ratio)

        # Split data
        train_data = qa_pairs[:train_idx]
        val_data = qa_pairs[train_idx:val_idx]
        test_data = qa_pairs[val_idx:]

        logger.info(
            f"Split data into {len(train_data)} training, {len(val_data)} validation, and {len(test_data)} test samples"
        )
        return train_data, val_data, test_data

    def format_for_llama3(
        self,
        train_data: List[Dict[str, str]],
        val_data: List[Dict[str, str]],
        test_data: List[Dict[str, str]],
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Format the data for Llama 3 fine-tuning.

        Args:
            train_data: List of training examples
            val_data: List of validation examples
            test_data: List of testing examples

        Returns:
            Dictionary with formatted data
        """
        logger.info("Formatting data for Llama 3 fine-tuning")

        def format_item(item):
            # Extract question from the original prompt
            question = item.get("instruction", "").strip()

            # Get the answer content, enhancing it if needed
            answer = item.get("response", "").strip()
            answer = self._enhance_answer_content(question, answer)

            # Format with Llama 3 chat template
            formatted_text = {
                "input": question,
                "output": answer,
                "text": f"<|system|>\nYou are a knowledgeable first responder assistant designed to provide helpful information about emergency procedures and protocols.\n<|user|>\n{question}\n<|assistant|>\n{answer}",
            }

            return formatted_text

        # Format each dataset
        formatted_train = [format_item(item) for item in train_data]
        formatted_val = [format_item(item) for item in val_data]
        formatted_test = [format_item(item) for item in test_data]

        return {
            "train": formatted_train,
            "validation": formatted_val,
            "test": formatted_test,
        }

    def _enhance_answer_content(self, question, answer):
        """
        Enhance answer content to ensure completeness and emphasize primary purposes.

        Args:
            question: The question string
            answer: The answer string

        Returns:
            Enhanced answer string
        """
        # Make sure all answers are complete sentences
        if not answer.strip().endswith((".", "!", "?")):
            answer = answer.strip() + "."

        # If answer is too short, try to enhance it based on topic
        if len(answer.split()) < 15:
            # For protective equipment questions
            if (
                "ppe" in question.lower()
                or "protective" in question.lower()
                or "equipment" in question.lower()
            ):
                if "protect" not in answer.lower() and "hazard" not in answer.lower():
                    answer += " Its primary purpose is to protect first responders from various hazards and minimize the risk of injury or fatality."

            # For procedure-related questions
            if (
                "procedure" in question.lower()
                or "protocol" in question.lower()
                or "step" in question.lower()
            ):
                if "safety" not in answer.lower() and "ensure" not in answer.lower():
                    answer += " Following proper procedures ensures safety and effective emergency response."

            # For emergency response questions
            if (
                "emergency" in question.lower()
                or "response" in question.lower()
                or "incident" in question.lower()
            ):
                if "critical" not in answer.lower() and "time" not in answer.lower():
                    answer += " Time-critical decisions during emergency response require proper training and adherence to protocols."

            # For medical-related questions
            if (
                "medical" in question.lower()
                or "treatment" in question.lower()
                or "patient" in question.lower()
            ):
                if "care" not in answer.lower() and "assessment" not in answer.lower():
                    answer += " Proper patient assessment and care are fundamental to medical emergency response."

        return answer

    def save_dataset(self, dataset: Dict[str, List[Dict[str, str]]]) -> None:
        """
        Save dataset to a JSON file.

        Args:
            dataset: Dictionary with formatted data
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.output_file.parent, exist_ok=True)

        # Save to JSON file
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)

        logger.info(
            f"Saved dataset with {len(dataset['train'])} training examples to {self.output_file}"
        )

    def run(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Run the dataset creation pipeline.

        Returns:
            Dictionary with formatted data
        """
        logger.info("Starting dataset creation...")

        # Load preprocessed data
        documents = self.load_preprocessed_data()

        # Filter documents
        filtered_docs = self._filter_documents(documents)

        # Generate QA pairs
        qa_pairs = self.generate_question_answer_pairs(filtered_docs)

        # Filter QA pairs
        filtered_qa_pairs = self._filter_qa_pairs(qa_pairs)

        # Split data
        logger.info("Splitting data into train/validation/test sets")
        train_data, val_data, test_data = self.split_data(filtered_qa_pairs)

        # Format data for Llama 3
        dataset = self.format_for_llama3(train_data, val_data, test_data)

        # Save dataset
        logger.info("Saving the final dataset")
        self.save_dataset(dataset)

        logger.info("Dataset creation completed successfully!")
        return dataset

    def _format_data_for_llama3(self, examples):
        """
        Format examples specifically for Llama 3 chat model.

        Args:
            examples: List of examples to format

        Returns:
            List of formatted examples
        """
        formatted_examples = []

        # Llama 3 uses a specific format with <|system|> <|user|> <|assistant|>
        for example in examples:
            if "instruction" in example and "response" in example:
                # Format instruction and response using Llama 3 chat format
                formatted_example = {
                    "text": f"<|system|>\nYou are a knowledgeable first responder assistant designed to provide helpful information about emergency procedures and protocols.\n<|user|>\n{example['instruction'].strip()}\n<|assistant|>\n{example['response'].strip()}",
                    "instruction": example["instruction"],
                    "response": example["response"],
                }

                # Add metadata if available
                if "metadata" in example:
                    formatted_example["metadata"] = example["metadata"]

                formatted_examples.append(formatted_example)

        return formatted_examples
