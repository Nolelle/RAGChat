"""
Dataset creation module for the FirstRespondersChatbot project.

This module provides functionality to generate a pseudo-supervised dataset
for fine-tuning the TinyLlama model.
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
        similarity_threshold: float = 0.8,  # Threshold for duplicate detection
        max_examples_per_doc: int = 3,  # Maximum examples to generate from a document
        model_format: str = "tinyllama",  # Options: "tinyllama" or "llama"
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
            model_format: Format to use for the dataset (tinyllama or llama)
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

        # Question templates by type
        question_templates = {
            "factual": [
                "What information can you provide about {}?",
                "What are the key points about {}?",
                "What does the manual say about {}?",
                "What is the primary purpose of {}?",
                "What is the core function of {} in emergency response?",
            ],
            "procedural": [
                "What procedures are involved with {}?",
                "How should first responders handle situations involving {}?",
                "What are the steps for {}?",
                "What is the proper protocol for {}?",
                "How do you ensure {} is properly maintained while keeping it ready for its primary protective purpose?",
            ],
            "explanatory": [
                "Can you explain the concept of {} in simple terms?",
                "How would you describe {} to a new first responder?",
                "Explain {} as if I'm a trainee first responder.",
                "What specific hazards does {} protect against?",
                "How does {} minimize risk of injury or fatality?",
            ],
            "application": [
                "How is {} applied in emergency situations?",
                "What are the best practices for {} in emergency situations?",
                "As a first responder, what should I know about {}?",
            ],
        }

        # Better contextual instruction prompts that match inference
        instruction_templates = [
            "Answer the question based on the following context. You are a first responders chatbot designed to help with training and education.",
            "You are a first responder education assistant. Answer the question based ONLY on the provided context.",
            "As a first responder training assistant, use the provided context to answer the question.",
            "Based on the context provided, answer this first responder training question.",
        ]

        # Group similar documents to create synthetic context
        logger.info("Starting to group similar documents")
        grouped_docs = self._group_similar_documents(filtered_documents)
        logger.info(
            f"Created {len(grouped_docs)} document groups for context generation"
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

            # Extract potential topics from keywords
            keywords = self._extract_keywords(primary_content, top_n=5)

            # Generate examples based on each keyword topic
            for keyword in keywords:
                if keyword in generated_topics:
                    continue

                # Add to tracking set
                generated_topics.add(keyword)

                # Create context similar to inference time
                context_text = ""
                for i, doc in enumerate(doc_group[:3]):  # Use up to 3 docs per question
                    meta_info = ""
                    if "meta" in doc and doc["meta"]:
                        if "file_name" in doc["meta"]:
                            meta_info += f" [Source: {doc['meta']['file_name']}]"
                        if "page_number" in doc["meta"]:
                            meta_info += f" [Page: {doc['meta']['page_number']}]"

                    context_text += (
                        f"\n### Document {i+1}{meta_info}:\n{doc['content']}\n"
                    )

                # Choose question types based on content
                q_types = list(question_templates.keys())

                # Generate questions for different question types
                for q_type in random.sample(q_types, min(2, len(q_types))):
                    templates = question_templates[q_type]

                    # Generate the question
                    q_template = random.choice(templates)
                    question = q_template.format(keyword)

                    # Choose a random instruction template
                    instruction = random.choice(instruction_templates)

                    # Create the full prompt
                    prompt = f"""{instruction}

Context:
{context_text}

Question: {question}

Answer:"""

                    # Generate answer paraphrases for data augmentation
                    answer_prefixes = [
                        f"Based on the provided information, {keyword} is",
                        f"According to the documentation, {keyword} refers to",
                        f"The materials indicate that {keyword} involves",
                        f"From a first responder perspective, {keyword} means",
                        f"In emergency situations, {keyword} is defined as",
                    ]

                    # Get relevant content snippet
                    content_snippet = primary_content
                    if len(content_snippet) > 300:
                        content_snippet = content_snippet[:300] + "..."

                    # Create answer with a prefix for better structure
                    answer = f"{random.choice(answer_prefixes)} {content_snippet}"

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

    def format_for_tinyllama(
        self,
        train_data: List[Dict[str, str]],
        val_data: List[Dict[str, str]],
        test_data: List[Dict[str, str]],
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Format data for TinyLlama fine-tuning.

        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Test data

        Returns:
            Dictionary with formatted data
        """
        system_message = "You are a first responders chatbot designed to provide accurate information about emergency procedures and protocols based on official training materials."

        # Format data for TinyLlama with chat template
        def format_item(item):
            return {
                "input": f"{item['question']}",
                "output": f"{item['answer']}",
                "text": f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{item['question']} [/INST] {item['answer']}</s>",
            }

        formatted_data = {
            "train": [format_item(item) for item in train_data],
            "validation": [format_item(item) for item in val_data],
            "test": [format_item(item) for item in test_data],
        }

        logger.info("Formatted data for TinyLlama fine-tuning")
        return formatted_data

    def format_for_llama(
        self,
        train_data: List[Dict[str, str]],
        val_data: List[Dict[str, str]],
        test_data: List[Dict[str, str]],
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Format data for Llama 3.1 fine-tuning.

        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Test data

        Returns:
            Dictionary with formatted data
        """
        system_message = "You are a first responders chatbot designed to provide accurate information about emergency procedures and protocols based on official training materials. Always provide comprehensive, standalone answers that cover all relevant aspects of a topic. Emphasize primary purposes and functions first before discussing secondary details like maintenance. When discussing equipment or procedures, explain what specific hazards they address and how they help minimize risks. Use clear, authoritative language appropriate for first responder training."

        # Format data for Llama 3.1 with chat template
        def format_item(item):
            enhanced_answer = self._enhance_answer_content(
                item["question"], item["answer"]
            )
            return {
                "input": f"{item['question']}",
                "output": f"{enhanced_answer}",
                "text": f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{item['question']} [/INST] {enhanced_answer}</s>",
            }

        formatted_data = {
            "train": [format_item(item) for item in train_data],
            "validation": [format_item(item) for item in val_data],
            "test": [format_item(item) for item in test_data],
        }

        logger.info("Formatted data for Llama 3.1 fine-tuning")
        return formatted_data

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

        # Generate question-answer pairs
        logger.info("Starting question-answer pair generation process")
        qa_pairs = self.generate_question_answer_pairs(documents)
        logger.info(f"Successfully generated {len(qa_pairs)} question-answer pairs")

        # Split data
        logger.info("Splitting data into train/validation/test sets")
        train_data, val_data, test_data = self.split_data(qa_pairs)

        # Format data for the selected model
        logger.info(f"Formatting data for {self.model_format} fine-tuning")
        if self.model_format.lower() == "tinyllama":
            dataset = self.format_for_tinyllama(train_data, val_data, test_data)
        elif self.model_format.lower() == "llama":
            dataset = self.format_for_llama(train_data, val_data, test_data)
        else:
            # Default to TinyLlama format
            dataset = self.format_for_tinyllama(train_data, val_data, test_data)

        # Save dataset
        logger.info("Saving the final dataset")
        self.save_dataset(dataset)

        logger.info("Dataset creation completed successfully!")
        return dataset
