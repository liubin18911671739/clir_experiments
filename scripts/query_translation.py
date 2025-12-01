#!/usr/bin/env python3
"""
Cross-lingual query translation for CLIR.

Supports multiple translation services:
- Google Translate API
- Azure Translator
- Local translation models (MarianMT)

Usage:
    python scripts/query_translation.py --config config/neuclir.yaml \
        --topics data/topics/eng.topics.txt \
        --source_lang eng --target_lang fas \
        --service local

    python scripts/query_translation.py --config config/neuclir.yaml \
        --topics data/topics/eng.topics.txt \
        --source_lang eng --target_lang fas \
        --service google --api_key YOUR_API_KEY
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import os

from utils_io import load_yaml, ensure_dir, get_repo_root, resolve_path
from utils_topics import parse_trec_topics, write_trec_topics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QueryTranslator:
    """Base class for query translation services."""

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate text from source language to target language.

        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Translated text
        """
        raise NotImplementedError


class LocalTranslator(QueryTranslator):
    """
    Local translator using HuggingFace MarianMT models.

    No API key required, runs locally.
    """

    def __init__(self, device: str = 'cuda'):
        """
        Initialize local translator.

        Args:
            device: Device to use ('cuda' or 'cpu')
        """
        from transformers import MarianMTModel, MarianTokenizer
        import torch

        self.device = device if torch.cuda.is_available() else 'cpu'
        self.models = {}  # Cache models
        self.tokenizers = {}

        logger.info(f"Local translator initialized on {self.device}")

    def _get_model_name(self, source_lang: str, target_lang: str) -> str:
        """Get MarianMT model name for language pair."""
        # Map language codes to MarianMT format
        lang_map = {
            'eng': 'en',
            'fas': 'fa',
            'rus': 'ru',
            'zho': 'zh',
            'ara': 'ar',
            'fra': 'fr',
            'deu': 'de',
            'spa': 'es'
        }

        src = lang_map.get(source_lang, source_lang)
        tgt = lang_map.get(target_lang, target_lang)

        # Try direct model
        model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"

        # Fallback: use English as pivot
        if source_lang != 'eng' and target_lang != 'eng':
            logger.warning(
                f"Direct {source_lang}->{target_lang} model may not exist. "
                f"Consider using English as pivot language."
            )

        return model_name

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate using MarianMT."""
        from transformers import MarianMTModel, MarianTokenizer

        model_name = self._get_model_name(source_lang, target_lang)
        cache_key = f"{source_lang}_{target_lang}"

        # Load or get cached model
        if cache_key not in self.models:
            try:
                logger.info(f"Loading translation model: {model_name}")
                self.tokenizers[cache_key] = MarianTokenizer.from_pretrained(model_name)
                self.models[cache_key] = MarianMTModel.from_pretrained(model_name)
                self.models[cache_key] = self.models[cache_key].to(self.device)
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise

        tokenizer = self.tokenizers[cache_key]
        model = self.models[cache_key]

        # Translate
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs)

        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated


class GoogleTranslator(QueryTranslator):
    """Google Cloud Translation API wrapper."""

    def __init__(self, api_key: str):
        """
        Initialize Google Translator.

        Args:
            api_key: Google Cloud API key
        """
        try:
            from google.cloud import translate_v2 as translate
            import google.auth
        except ImportError:
            raise ImportError(
                "Google Cloud Translation requires: pip install google-cloud-translate"
            )

        # Set credentials
        if api_key:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = api_key

        self.client = translate.Client()
        logger.info("Google Translator initialized")

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate using Google Cloud Translation API."""
        result = self.client.translate(
            text,
            source_language=source_lang,
            target_language=target_lang
        )
        return result['translatedText']


class AzureTranslator(QueryTranslator):
    """Azure Cognitive Services Translator wrapper."""

    def __init__(self, api_key: str, region: str = 'global'):
        """
        Initialize Azure Translator.

        Args:
            api_key: Azure Translator API key
            region: Azure region (default: 'global')
        """
        import requests

        self.api_key = api_key
        self.region = region
        self.endpoint = 'https://api.cognitive.microsofttranslator.com'
        logger.info("Azure Translator initialized")

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate using Azure Translator API."""
        import requests
        import uuid

        path = '/translate?api-version=3.0'
        params = f'&from={source_lang}&to={target_lang}'
        constructed_url = self.endpoint + path + params

        headers = {
            'Ocp-Apim-Subscription-Key': self.api_key,
            'Ocp-Apim-Subscription-Region': self.region,
            'Content-type': 'application/json',
            'X-ClientTraceId': str(uuid.uuid4())
        }

        body = [{'text': text}]

        response = requests.post(constructed_url, headers=headers, json=body)
        response.raise_for_status()

        result = response.json()
        return result[0]['translations'][0]['text']


def translate_topics(
    topics_path: str,
    output_path: str,
    source_lang: str,
    target_lang: str,
    service: str = 'local',
    api_key: Optional[str] = None,
    device: str = 'cuda'
) -> None:
    """
    Translate topic file from source language to target language.

    Args:
        topics_path: Path to source topics file
        output_path: Path to output translated topics
        source_lang: Source language code
        target_lang: Target language code
        service: Translation service ('local', 'google', 'azure')
        api_key: API key for cloud services
        device: Device for local translation
    """
    # Load source topics
    logger.info(f"Loading topics from: {topics_path}")
    topics = parse_trec_topics(topics_path)
    logger.info(f"Loaded {len(topics)} topics")

    # Initialize translator
    if service == 'local':
        import torch
        translator = LocalTranslator(device=device)
    elif service == 'google':
        if not api_key:
            raise ValueError("Google Translator requires --api_key")
        translator = GoogleTranslator(api_key)
    elif service == 'azure':
        if not api_key:
            raise ValueError("Azure Translator requires --api_key")
        translator = AzureTranslator(api_key)
    else:
        raise ValueError(f"Unknown service: {service}")

    # Translate topics
    logger.info(f"Translating {source_lang} -> {target_lang} using {service}")
    translated_topics = {}

    for qid, query_text in topics.items():
        try:
            translated = translator.translate(query_text, source_lang, target_lang)
            translated_topics[qid] = translated

            logger.debug(f"Query {qid}:")
            logger.debug(f"  Original ({source_lang}): {query_text}")
            logger.debug(f"  Translated ({target_lang}): {translated}")

        except Exception as e:
            logger.error(f"Failed to translate query {qid}: {e}")
            # Fallback to original
            translated_topics[qid] = query_text

    # Write translated topics
    ensure_dir(Path(output_path).parent)
    logger.info(f"Writing translated topics to: {output_path}")
    write_trec_topics(translated_topics, output_path, format='simple')

    logger.info(f"Translation complete. {len(translated_topics)} topics saved.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Cross-lingual query translation"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--topics',
        type=str,
        required=True,
        help='Path to source topics file'
    )
    parser.add_argument(
        '--source_lang',
        type=str,
        required=True,
        help='Source language code (e.g., eng, fas, rus)'
    )
    parser.add_argument(
        '--target_lang',
        type=str,
        required=True,
        help='Target language code (e.g., eng, fas, rus)'
    )
    parser.add_argument(
        '--service',
        type=str,
        choices=['local', 'google', 'azure'],
        default='local',
        help='Translation service (default: local)'
    )
    parser.add_argument(
        '--api_key',
        type=str,
        default=None,
        help='API key for cloud translation services'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for translated topics'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device for local translation (default: cuda)'
    )

    args = parser.parse_args()

    # Generate output path if not specified
    if args.output is None:
        topics_path = Path(args.topics)
        output_path = topics_path.parent / f"{topics_path.stem}_{args.target_lang}.topics.txt"
    else:
        output_path = args.output

    # Translate topics
    translate_topics(
        args.topics,
        str(output_path),
        args.source_lang,
        args.target_lang,
        service=args.service,
        api_key=args.api_key,
        device=args.device
    )

    logger.info("Query translation complete!")


if __name__ == '__main__':
    main()
