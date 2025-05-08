from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, collect_list, struct
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, TimestampType
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SocialLinkDataProcessor:
    def __init__(self, spark_master: str = "local[*]"):
        self.spark = SparkSession.builder \
            .appName("SocialLinkDataProcessor") \
            .master(spark_master) \
            .config("spark.executor.memory", "4g") \
            .config("spark.driver.memory", "2g") \
            .getOrCreate()
        
        # Define schemas
        self.interaction_schema = StructType([
            StructField("user_id", IntegerType(), False),
            StructField("item_id", IntegerType(), False),
            StructField("timestamp", TimestampType(), False),
            StructField("interaction_type", StringType(), False)
        ])
        
    def process_interactions(self, input_path: str, output_path: str):
        """Process user-item interactions and generate training data."""
        logger.info(f"Processing interactions from {input_path}")
        
        # Read interactions
        interactions_df = self.spark.read \
            .schema(self.interaction_schema) \
            .parquet(input_path)
        
        # Generate positive samples
        positive_samples = interactions_df \
            .filter(col("interaction_type").isin(["click", "connect"])) \
            .select("user_id", "item_id") \
            .distinct()
        
        # Generate negative samples
        all_items = interactions_df.select("item_id").distinct()
        user_items = interactions_df.select("user_id", "item_id").distinct()
        
        negative_samples = user_items \
            .crossJoin(all_items) \
            .join(positive_samples, ["user_id", "item_id"], "left_anti") \
            .select("user_id", "item_id")
        
        # Combine positive and negative samples
        training_data = positive_samples \
            .withColumn("label", col("1")) \
            .union(negative_samples.withColumn("label", col("0")))
        
        # Save processed data
        training_data.write \
            .mode("overwrite") \
            .parquet(output_path)
        
        logger.info(f"Processed data saved to {output_path}")
        
    def generate_user_features(self, input_path: str, output_path: str):
        """Generate user features from interaction history."""
        logger.info(f"Generating user features from {input_path}")
        
        interactions_df = self.spark.read \
            .schema(self.interaction_schema) \
            .parquet(input_path)
        
        # Calculate user activity metrics
        user_features = interactions_df \
            .groupBy("user_id") \
            .agg(
                collect_list(struct("item_id", "interaction_type", "timestamp")).alias("interaction_history"),
                count("*").alias("total_interactions")
            )
        
        # Save user features
        user_features.write \
            .mode("overwrite") \
            .parquet(output_path)
        
        logger.info(f"User features saved to {output_path}")
        
    def generate_item_features(self, input_path: str, output_path: str):
        """Generate item features from interaction history."""
        logger.info(f"Generating item features from {input_path}")
        
        interactions_df = self.spark.read \
            .schema(self.interaction_schema) \
            .parquet(input_path)
        
        # Calculate item popularity metrics
        item_features = interactions_df \
            .groupBy("item_id") \
            .agg(
                count("*").alias("total_interactions"),
                countDistinct("user_id").alias("unique_users")
            )
        
        # Save item features
        item_features.write \
            .mode("overwrite") \
            .parquet(output_path)
        
        logger.info(f"Item features saved to {output_path}")
        
    def close(self):
        """Close Spark session."""
        self.spark.stop() 