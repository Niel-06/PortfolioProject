# Below dependencies needed to run this code
# pip install google-genai
# pip install openpyxl
# pip install pandas
#
# This script aims to automate the prioritization and categorization of support tickets using a multi-faceted logic model. The solution integrates three key components:
#   
#   Core logic
#   - Rule-Based System: defined set of rules will classify and prioritize tickets based on established fields like the enquiry classification (category) and sentiment  #        level of the content of the ticket. This also look for any tag that used by the rule.
#   - Historical Pattern Insight: The system will analyze historical or recurring ticket patterns. For instance, if a common, non-critical issue is identified, its priority #    will be automatically lowered, streamlining the queue.
#   - AI-Driven Customer Sentiment: Customer sentiment will be analyzed using a language model (LLM) to generate a confidence score. This score, derived from the ticket's #      subject or body text, will inform priority, ensuring urgent or highly frustrated customers are addressed quickly.
#   
#   Key Assumptions
#   The successful implementation of this model is based on the following assumptions regarding data and fields:
#   - A specific, consistent field (e.g., enquiry classification) is available and used as the primary source for ticket categorization. 
#   - The raw ticket data will be provided in a structured format, specifically an Excel file, which the script and LLM will directly process.
#   - The LLM is capable of accurately extracting sentiment from the ticket's subject or body text.
#   
#   Scalability, Limitations and Trade-Offs   
#    - The primary trade-off for this, line-by-line analysis is processing time. Analyzing a large dataset sequentially, ticket by ticket, will result in longer execution times. Additionally, the overall limitation is dependent on the constraints of the chosen LLM, specifically its token limit, which could restrict the volume of text analyzed per ticket or the batch size for processing.
#   - The script is designed for future-proofing and easy management. The prioritization and categorization scoring logic will be externalized and easily manageable, allowing  for quick adjustments without modifying core code.
#   - While the current design is sequential, the system allows for a future transition to batch processing to significantly improve performance and handle higher volumes.


#import all necessart dependency, using pandas to read via excel
import google.genai as genai
import json
import os
import pandas as pd
import logging
from datetime import datetime
from typing import Optional, Dict, Tuple
import openpyxl

# Initialize logging to capture issue
def setup_logging(log_file: str = None):
    """Setup logging configuration."""
    if log_file is None:
        log_file = f"ticket_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Initialize the Gemini Client using API KEY
try:
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY")
    )
    logger.info("Gemini client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini client: {str(e)}")
    raise

# JSON Schema for structured output
TICKET_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {
            "type": "string",
            "description": "The dominant emotional state: negative, positive, neutral, or mixed."
        },
        "confidence_score": {
            "type": "number",
            "description": "A float from 0.0 to 1.0 reflecting the certainty of high urgency/priority based on the text."
        },
        "detected_tags": {
            "type": "array",
            "description": "A list of all relevant tags found, including mandatory ones (frustrated, error, issue, product, exam) and any new, relevant urgency-related keywords.",
            "items": {
                "type": "string"
            }
        },
        "is_recurring": {
            "type": "boolean",
            "description": "True if the ticket text indicates this is a recurring issue (phrases like 'again', 'still not working', 'same problem', 'multiple times', etc.)"
        }
    },
    "required": ["sentiment", "confidence_score", "detected_tags", "is_recurring"]
}

# System Instruction to be used by the LLM 
SYSTEM_INSTRUCTION = (
    "You are a specialized AI designed for immediate ticket triage. Your sole function is to "
    "analyze the user's ticket text for urgency, emotional state, and relevant keywords. "
    "Detect if the ticket indicates a recurring issue by looking for phrases like 'again', 'still', "
    "'same problem', 'multiple times', 'keeps happening', etc. "
    "Output a clean, consistent JSON object that strictly adheres to the provided schema. "
    "The confidence_score must reflect your certainty of the ticket's high urgency, from 0.0 (low certainty) to 1.0 (high certainty)."
)

# Configuration for priority scoring
# this can be managed manually in case there is some changes on the pattern
DEFAULT_CONFIG = {
    "CATEGORY_SCORES": {
        "Escalation": 30,
        "System Issue": 30,
        "Billing": 20,
        "Results": 10,
        "Account": 0,
        "Suggestion": 0,
        "General Inquiry": 0
    },
    "SENTIMENT_WEIGHTS": {
        "negative": 20,
        "mixed": 10,
        "neutral": 0,
        "positive": -10
    },
    "TAG_SCORES": {
        "frustrated": 15,
        "error": 20,
        "issue": 10,
        "product": 5,
        "exam": 25,
        "urgent": 30,
        "broken": 20,
        "NEW_TAG_BONUS": 8
    },
    "HISTORICAL_RULES": {
        "RECURRENCE_PENALTY": 15,
        "FOLLOW_UP_THRESHOLD": 2,
        "FOLLOW_UP_BONUS": 12
    },
    "LLM_CONFIDENCE_SCALER": 50,
    "TIERS": {
        "URGENT_MIN": 80,
        "HIGH_MIN": 50,
        "MEDIUM_MIN": 25
    }
}

#This function should used for connecting to the LLM and will analyze tickets description and also identify new tags
# Args: ticket_description: the ticket to analyze
# max_retries: Max number of retry attemps on failure
def analyze_ticket_with_gemini(ticket_description: str, max_retries: int = 3) -> Optional[Dict]:
    
    prompt = f"Ticket Description for Analysis:\n\n---\n{ticket_description}\n---"
    
    #loop to connect within the LLM
    for attempt in range(max_retries):
        try:
            logger.info(f"Analyzing ticket (attempt {attempt + 1}/{max_retries})")
            
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config={
                    "system_instruction": SYSTEM_INSTRUCTION,
                    "response_mime_type": "application/json",
                    "response_schema": TICKET_SCHEMA,
                }
            )
            
            # Log token usage if available
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                logger.info(f"Token usage - Prompt: {usage.prompt_token_count}, "
                          f"Response: {usage.candidates_token_count}, "
                          f"Total: {usage.total_token_count}")
            
            result = json.loads(response.text)
            logger.info(f"Successfully analyzed ticket - Sentiment: {result.get('sentiment')}, "
                       f"Confidence: {result.get('confidence_score'):.2f}")
            
            return result

        # Error handling layer if there's some issue on the LLM or the Excel    
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                logger.error(f"Failed to parse response after {max_retries} attempts")
                return None
                
        except Exception as e:
            logger.error(f"API error on attempt {attempt + 1}: {type(e).__name__} - {str(e)}")
            if attempt == max_retries - 1:
                logger.error(f"Failed to analyze ticket after {max_retries} attempts")
                return None
            
            # Wait before retry (exponential backoff)
            import time
            wait_time = 2 ** attempt
            logger.info(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    
    return None

# This function should calculate the priority score based on the rule we set on the schema and also on the AI output
# This return tuples of (tier, total_score, score_breakdown)
def calculate_priority_score(ai_data: dict, ticket_metadata: dict, config: dict) -> Tuple[str, float, dict]:
    """
    Calculates the final total priority score based on all rules and AI output.
    
    Returns:
        Tuple of (tier, total_score, score_breakdown)
    """
    #try exception to capture issue on the run.
    try:
        total_score = 0
        score_breakdown = {}
        
        # 1. Rule-Based Category Score (R_cat), this is based on our default config, which we supply the value of category
        category = ticket_metadata.get("Category", "General Inquiry")
        category_score = config["CATEGORY_SCORES"].get(category, 0)
        total_score += category_score
        score_breakdown['category'] = category_score
        logger.debug(f"Category score ({category}): +{category_score}")
        
        # 2. Historical/Pattern Score (R_hist), this is to check if the ticket is_recurring or always been following up with customer
        historical_score = 0
        if ticket_metadata.get("is_recurring"):
            historical_score += config["HISTORICAL_RULES"]["RECURRENCE_PENALTY"]
            logger.debug(f"Recurring issue penalty: +{config['HISTORICAL_RULES']['RECURRENCE_PENALTY']}")
        
        if ticket_metadata.get("follow_up_count", 0) >= config["HISTORICAL_RULES"]["FOLLOW_UP_THRESHOLD"]:
            historical_score += config["HISTORICAL_RULES"]["FOLLOW_UP_BONUS"]
            logger.debug(f"Follow-up bonus: +{config['HISTORICAL_RULES']['FOLLOW_UP_BONUS']}")
        
        total_score += historical_score
        score_breakdown['historical'] = historical_score

        # 3. Sentiment Weight (W_sent), this will be the score of the sentiment that was generated by AI this will add the score to the overall score
        sentiment = ai_data.get("sentiment", "neutral")
        sentiment_weight = config["SENTIMENT_WEIGHTS"].get(sentiment, 0)
        total_score += sentiment_weight
        score_breakdown['sentiment'] = sentiment_weight
        logger.debug(f"Sentiment weight ({sentiment}): +{sentiment_weight}")
        
        # 4. Explicit Tag Score (R_tag), this will score the tags identify on the description and also check for new tag that can be learned based on the tickets
        tag_score = 0
        new_learned_tag_bonus_applied = False
        detected_tags = ai_data.get("detected_tags", [])
        
        for tag in detected_tags:
            tag_score += config["TAG_SCORES"].get(tag, 0)
            
            if tag not in config["TAG_SCORES"]:
                if sentiment in ["negative", "mixed"] and not new_learned_tag_bonus_applied:
                    tag_score += config["TAG_SCORES"]["NEW_TAG_BONUS"]
                    new_learned_tag_bonus_applied = True
                    logger.debug(f"New tag bonus applied for: {tag}")
            
        total_score += tag_score
        score_breakdown['tags'] = tag_score
        logger.debug(f"Tag score: +{tag_score}")
        
        #  5. Confidence Bonus (C * Scaler) - this will just convert the AI score since it will be only 0.0 or 0.1, since it will not add or affect the scoring so we need to scale it so it will have impact on the overall score
        confidence = ai_data.get("confidence_score", 0)
        confidence_bonus = confidence * config["LLM_CONFIDENCE_SCALER"]
        total_score += confidence_bonus
        score_breakdown['confidence'] = round(confidence_bonus, 2)
        logger.debug(f"Confidence bonus ({confidence:.2f} * 50): +{confidence_bonus:.2f}")

        final_score = round(total_score, 2)
        
        # 6. Final Tier Assignment - this will assign the tier based on the overall score calculated on the above scoring
        if final_score >= config["TIERS"]["URGENT_MIN"]:
            tier = "URGENT"
        elif final_score >= config["TIERS"]["HIGH_MIN"]:
            tier = "HIGH"
        elif final_score >= config["TIERS"]["MEDIUM_MIN"]:
            tier = "MEDIUM"
        else:
            tier = "LOW"
        
        logger.info(f"Final score: {final_score} → Tier: {tier}")
        
        return tier, final_score, score_breakdown
        
    except Exception as e:
        logger.error(f"Error calculating priority score: {str(e)}")
        # Return default values on error
        return "LOW", 0.0, {"category": 0, "historical": 0, "sentiment": 0, "tags": 0, "confidence": 0}

# this function reads ticket description from excel and analyze them with priority scoring and also add the result of AI and rule based
#    Args:
#        excel_file_path: Path to the Excel file
#        description_column: Name of the column containing ticket descriptions
#        category_column: Name of the column containing ticket categories
#        ticket_id_column: Name of the column containing ticket IDs (for follow-up detection)
#        config: Configuration dict for scoring (uses DEFAULT_CONFIG if None)
#        output_file: Optional path to save results as Excel

def process_tickets_from_excel(excel_file_path: str, 
                               description_column: str = "description",
                               category_column: str = "Category",
                               ticket_id_column: str = "ticket_id",
                               config: dict = None,
                               output_file: str = None):
    
    if config is None:
        config = DEFAULT_CONFIG
    
    # Track metrics
    metrics = {
        "total_tickets": 0,
        "successful": 0,
        "failed": 0,
        "skipped": 0,
        "total_tokens": 0,
        "start_time": datetime.now()
    }
    
    try:
        # Read the Excel file
        logger.info(f"Reading Excel file: {excel_file_path}")
        df = pd.read_excel(excel_file_path)
        logger.info(f"Loaded {len(df)} rows from Excel")
        
        # Check if the description column exist
        if description_column not in df.columns:
            error_msg = f"Column '{description_column}' not found. Available columns: {list(df.columns)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Calculate follow-up counts based on ticket ID (if they have same ticket ID)
        follow_up_counts = {}
        if ticket_id_column in df.columns:
            follow_up_counts = df[ticket_id_column].value_counts().to_dict()
            logger.info(f"Calculated follow-up counts for {len(follow_up_counts)} unique ticket IDs")
        else:
            logger.warning(f"'{ticket_id_column}' column not found. Follow-up detection disabled.")
        
        # Initialize result columns within the dataframe
        df['sentiment'] = None
        df['confidence_score'] = None
        df['detected_tags'] = None
        df['is_recurring'] = None
        df['follow_up_count'] = None
        df['priority_tier'] = None
        df['total_score'] = None
        df['score_category'] = None
        df['score_historical'] = None
        df['score_sentiment'] = None
        df['score_tags'] = None
        df['score_confidence'] = None
        df['processing_status'] = None
        
        # Process each ticket
        metrics["total_tickets"] = len(df)
        logger.info(f"Starting to process {metrics['total_tickets']} tickets...")
        
        #iterate within the dataframe
        for idx, row in df.iterrows():
            ticket_text = row[description_column]
            
            # Skip empty descriptions
            if pd.isna(ticket_text) or str(ticket_text).strip() == "":
                logger.warning(f"Row {idx + 1}: Skipping empty description")
                df.at[idx, 'processing_status'] = "SKIPPED_EMPTY"
                metrics["skipped"] += 1
                continue
            
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing Ticket {idx + 1}/{len(df)}")
                logger.info(f"{'='*60}")
                
                # Analyze the ticket with AI
                ai_result = analyze_ticket_with_gemini(str(ticket_text))
                
                if ai_result is None:
                    logger.error(f"Row {idx + 1}: Failed to get AI analysis")
                    df.at[idx, 'processing_status'] = "FAILED_AI_ANALYSIS"
                    metrics["failed"] += 1
                    continue
                
                # Get follow-up count from the ID to identify if they are also recurring ticket
                ticket_id = row.get(ticket_id_column) if ticket_id_column in df.columns else None
                follow_up_count = follow_up_counts.get(ticket_id, 1) - 1 if ticket_id else 0
                
                # Prepare metadata
                ticket_metadata = {
                    "Category": row.get(category_column, "General Inquiry") if category_column in df.columns else "General Inquiry",
                    "is_recurring": ai_result.get("is_recurring", False),
                    "follow_up_count": follow_up_count
                }
                
                # Calculate priority score
                tier, total_score, score_breakdown = calculate_priority_score(
                    ai_result, 
                    ticket_metadata, 
                    config
                )
                
                # Store results within dataframe
                df.at[idx, 'sentiment'] = ai_result.get('sentiment')
                df.at[idx, 'confidence_score'] = ai_result.get('confidence_score')
                df.at[idx, 'detected_tags'] = ', '.join(ai_result.get('detected_tags', []))
                df.at[idx, 'is_recurring'] = ai_result.get('is_recurring', False)
                df.at[idx, 'follow_up_count'] = follow_up_count
                df.at[idx, 'priority_tier'] = tier
                df.at[idx, 'total_score'] = total_score
                df.at[idx, 'score_category'] = score_breakdown.get('category', 0)
                df.at[idx, 'score_historical'] = score_breakdown.get('historical', 0)
                df.at[idx, 'score_sentiment'] = score_breakdown.get('sentiment', 0)
                df.at[idx, 'score_tags'] = score_breakdown.get('tags', 0)
                df.at[idx, 'score_confidence'] = score_breakdown.get('confidence', 0)
                df.at[idx, 'processing_status'] = "SUCCESS"
                
                metrics["successful"] += 1
                logger.info(f"✓ Row {idx + 1}: Successfully processed - Tier: {tier}")
                
            except Exception as e:
                logger.error(f"✗ Row {idx + 1}: Unexpected error - {type(e).__name__}: {str(e)}", exc_info=True)
                df.at[idx, 'processing_status'] = f"ERROR_{type(e).__name__}"
                metrics["failed"] += 1
                continue
        
        # Calculate processing time
        metrics["end_time"] = datetime.now()
        metrics["duration"] = (metrics["end_time"] - metrics["start_time"]).total_seconds()
        
        # Save results to a seperate excel file
        if output_file is None:
            output_file = excel_file_path.replace('.xlsx', '_analyzed.xlsx')
        
        df.to_excel(output_file, index=False)
        logger.info(f"\n{'='*60}")
        logger.info(f"✓ Results saved to: {output_file}")
        logger.info(f"{'='*60}")
        
        # Log final metrics
        logger.info("\n=== PROCESSING METRICS ===")
        logger.info(f"Total tickets: {metrics['total_tickets']}")
        logger.info(f"Successfully processed: {metrics['successful']}")
        logger.info(f"Failed: {metrics['failed']}")
        logger.info(f"Skipped: {metrics['skipped']}")
        logger.info(f"Duration: {metrics['duration']:.2f} seconds")
        logger.info(f"Average time per ticket: {metrics['duration']/max(metrics['total_tickets'], 1):.2f} seconds")
        
        return df
        
    except Exception as e:
        logger.error(f"Fatal error processing Excel file: {type(e).__name__} - {str(e)}", exc_info=True)
        raise



if __name__ == "__main__":
    # this will be the path of the excel which should have ticket data
    excel_path = "ticket_triage_test.xlsx"
    
    # Process tickets based on function
    results = process_tickets_from_excel(
        excel_file_path=excel_path,
        description_column="description",  # Change to match your column
        category_column="enq_classification",  # Change to match your column
        ticket_id_column="ticket_id",  # Change to match your column
        output_file="tickets_triage_analyzed.xlsx"
    )
    
    # Display summary
    print("\n=== Analysis Summary ===")
    print(f"Total tickets: {len(results)}")
    print(f"\nPriority Distribution:")
    print(results['priority_tier'].value_counts().to_dict())
    print(f"\nSentiment Distribution:")
    print(results['sentiment'].value_counts().to_dict())
    print(f"\nAverage Scores:")
    print(f"  Total Score: {results['total_score'].mean():.2f}")
    print(f"  Confidence: {results['confidence_score'].mean():.2f}")