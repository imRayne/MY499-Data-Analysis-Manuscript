import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import re
import jieba
import jieba.posseg as pseg
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from itertools import product
# Try to import gensim for standard coherence calculation
try:
    from gensim.models import CoherenceModel
    from gensim.corpora import Dictionary
    GENSIM_AVAILABLE = True
    print("✅ Gensim imported successfully, will use standard c_v coherence")
except ImportError:
    GENSIM_AVAILABLE = False
    print("⚠️ Gensim not available, will use simplified coherence calculation")

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class OptimizedLDAAnalyzer:
    def __init__(self):
        self.custom_stopwords = self.load_stopwords('cn_stopwords.txt')
        self.feminism_keywords = self.load_keywords('feminism keywords.txt')
    
    def load_stopwords(self, filename):
        """Load stopwords from file"""
        stopwords = set()
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word and not word.startswith('#'):
                        stopwords.add(word)
            print(f"✅ Successfully loaded {len(stopwords)} stopwords")
        except FileNotFoundError:
            print(f"⚠️ Stopwords file {filename} not found, using default stopwords")
            stopwords = {'的', '是', '在', '和', '有', '了', '个', '这', '那', '就', '不', '也', '我', '你', '他', '她'}
        return stopwords
    
    def load_keywords(self, filename):
        """Load keywords from file"""
        keywords = set()
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word and not word.startswith('#'):
                        keywords.add(word)
            print(f"✅ Successfully loaded {len(keywords)} feminism keywords")
        except FileNotFoundError:
            print(f"⚠️ Keywords file {filename} not found, using default keywords")
            keywords = {'女性主义', '女权主义', '性别平等', '性别歧视'}
        return keywords
    
    def find_csv_files(self, directory="."):
        """Automatically detect CSV files containing contents"""
        csv_files = []
        all_csv = glob.glob(os.path.join(directory, "*.csv"))
        
        for file_path in all_csv:
            filename = os.path.basename(file_path)
            if ('contents' in filename.lower() and 'comments' not in filename.lower()):
                csv_files.append(file_path)
        
        if not csv_files:
            print("❌ No CSV files containing 'contents' found")
            return []
        
        csv_files.sort()
        print(f"✅ Found {len(csv_files)} contents files:")
        for file in csv_files:
            filename = os.path.basename(file)
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
            date_info = f" (Date: {date_match.group(1)})" if date_match else ""
            print(f"  📄 {filename}{date_info}")
        
        return csv_files
    
    def load_data(self, file_paths=None, directory="."):
        """Load multiple CSV files and merge"""
        if file_paths is None:
            file_paths = self.find_csv_files(directory)
        
        if not file_paths:
            return pd.DataFrame()
        
        all_data = []
        for file_path in file_paths:
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                keyword = os.path.basename(file_path).replace('_contents', '').replace('.csv', '')
                df['keyword_source'] = keyword
                df['source_file'] = os.path.basename(file_path)
                all_data.append(df)
                print(f"✓ Loaded file {os.path.basename(file_path)}: {len(df)} records")
            except Exception as e:
                print(f"✗ Failed to load file {file_path}: {e}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"\n📊 Data merge completed: Total {len(combined_df)} records")
            return combined_df
        else:
            return pd.DataFrame()
    
    def clean_text(self, text):
        """Text cleaning, preserve Chinese characters"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Remove emoji, URLs, @usernames, #hashtags, [] tags
        emoji_pattern = re.compile("["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub('', text)
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'@[\w\u4e00-\u9fff]+', '', text)
        text = re.sub(r'#[^#]*#', '', text)
        text = re.sub(r'\[[^\]]*\]', '', text)
        
        # Preserve Chinese, English characters and basic punctuation
        text = re.sub(r'[^\u4e00-\u9fff\u3400-\u4dbfa-zA-Z\s，。！？、；：""''（）]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def segment_text(self, text):
        """Chinese word segmentation"""
        if not text:
            return []
        
        # Add professional vocabulary to jieba dictionary
        for word in self.feminism_keywords:
            jieba.add_word(word)
        
        # Use part-of-speech tagging
        words = pseg.cut(text)
        result = []
        valid_flags = ['n', 'v', 'a', 'nr', 'ns', 'nt', 'nz', 'vn', 'an', 'ng', 'nrt', 'nw']
        
        for word, flag in words:
            word = word.strip()
            if (len(word) >= 2 and 
                word not in self.custom_stopwords and 
                not re.match(r'^\d+$', word) and 
                (flag in valid_flags or word in self.feminism_keywords)):
                result.append(word)
        
        return result
    
    def preprocess_data(self, df):
        """Data preprocessing"""
        print("\n🔄 Starting data preprocessing...")
        
        # Remove duplicates
        if 'note_id' in df.columns:
            original_count = len(df)
            df = df.drop_duplicates(subset=['note_id'], keep='first')
            print(f"   Deduplication based on note_id: {original_count} → {len(df)} records")
        
        # Determine text column
        text_column = 'desc' if 'desc' in df.columns else 'content'
        if text_column not in df.columns:
            print("❌ Text column not found")
            return [], []
        
        print(f"📝 Using {text_column} column as analysis text")
        
        # Clean text
        print("   🧹 Cleaning text...")
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        
        # Filter empty text
        df = df[df['cleaned_text'].str.len() > 5]
        print(f"   After filtering short text: {len(df)} records")
        
        # Word segmentation
        print("   🔤 Performing Chinese word segmentation...")
        df['segmented_words'] = df['cleaned_text'].apply(self.segment_text)
        
        # Filter documents with too few words
        df = df[df['segmented_words'].apply(len) >= 3]
        print(f"   After filtering documents with too few words: {len(df)} records")
        
        # Extract text list
        texts = df['segmented_words'].tolist()
        
        # Show examples
        if len(texts) > 0:
            print(f"\n📝 Word segmentation examples:")
            for i, text in enumerate(texts[:3]):
                print(f"   Example{i+1}: {' '.join(text[:15])}...")
        
        print(f"✅ Data preprocessing completed, valid documents: {len(texts)}")
        return texts, df
    
    def create_word_frequency_analysis(self, texts):
        """Create word frequency analysis"""
        print("\n📊 Word frequency analysis...")
        
        # Count all words
        all_words = []
        for text in texts:
            all_words.extend(text)
        
        # Calculate word frequency
        word_freq = Counter(all_words)
        
        # Get high-frequency words
        top_words = word_freq.most_common(50)
        
        print("🔝 High-frequency words (top 20):")
        for i, (word, freq) in enumerate(top_words[:20]):
            print(f"   {i+1:2d}. {word}: {freq}")
        
        return word_freq, top_words
    
    def grid_search_alpha_beta(self, documents, num_topics, alpha_range=None, beta_range=None):
        """Grid search for optimal alpha and beta parameters"""
        if alpha_range is None:
            alpha_range = [0.01, 0.1, 0.5, 1.0]
        if beta_range is None:
            beta_range = [0.01, 0.1, 0.5, 1.0]
        
        print(f"\n🔍 Grid search for optimal alpha and beta parameters...")
        print(f"   Alpha range: {alpha_range}")
        print(f"   Beta range: {beta_range}")
        print(f"   Number of topics: {num_topics}")
        
        best_coherence = 0.0
        best_alpha = None
        best_beta = None
        best_model = None
        best_vectorizer = None
        
        # Prepare documents for coherence calculation
        tokenized_docs = []
        for doc in documents:
            tokens = doc.split()
            tokenized_docs.append(tokens)
        
        # Create word frequency dictionary
        word_freq = Counter()
        for doc_tokens in tokenized_docs:
            word_freq.update(doc_tokens)
        
        total_combinations = len(alpha_range) * len(beta_range)
        current_combination = 0
        
        for alpha, beta in product(alpha_range, beta_range):
            current_combination += 1
            print(f"   Progress: {current_combination}/{total_combinations} - Testing alpha={alpha}, beta={beta}")
            
            try:
                # TF-IDF vectorization
                vectorizer = TfidfVectorizer(
                    max_features=1000,
                    min_df=2,
                    max_df=0.5,
                    token_pattern=r'\b\w+\b',
                    sublinear_tf=True,
                    smooth_idf=True
                )
                
                doc_term_matrix = vectorizer.fit_transform(documents)
                
                lda_model = LatentDirichletAllocation(
                    n_components=num_topics,
                    random_state=42,
                    max_iter=100,
                    learning_method='online',
                    learning_offset=50.0,
                    doc_topic_prior=alpha,
                    topic_word_prior=beta,
                    batch_size=512,
                    evaluate_every=5,
                    perp_tol=1e-1,
                    mean_change_tol=1e-3
                )
                
                lda_model.fit(doc_term_matrix)
                perplexity = lda_model.perplexity(doc_term_matrix)
                
                # Calculate coherence
                feature_names = vectorizer.get_feature_names_out()
                coherence = self.calculate_simplified_coherence(lda_model, feature_names, tokenized_docs, word_freq)
                
                print(f"      Perplexity: {perplexity:.4f}, Coherence: {coherence:.4f}")
                
                # Choose best model based on coherence
                if coherence > best_coherence:
                    best_coherence = coherence
                    best_alpha = alpha
                    best_beta = beta
                    best_model = lda_model
                    best_vectorizer = vectorizer
                    print(f"      🎯 New best! Alpha={alpha}, Beta={beta}, Perplexity={perplexity:.4f}, Coherence={coherence:.4f}")
                
            except Exception as e:
                print(f"      ❌ Failed with alpha={alpha}, beta={beta}: {e}")
                continue
        
        print(f"\n🏆 Best parameters found:")
        print(f"   Alpha: {best_alpha}")
        print(f"   Beta: {best_beta}")
        print(f"   Coherence: {best_coherence:.4f}")
        
        return best_model, best_vectorizer, best_alpha, best_beta, []
    
    def compute_coherence_values(self, documents, start=2, limit=16, step=1):
        """Compute coherence and perplexity for different numbers of topics"""
        print(f"\n🔍 Hyperparameter search: Computing topic coherence (topic range: {start}-{limit-1})...")
        
        coherence_values = []
        perplexity_values = []
        model_list = []
        vectorizer_list = []
        
        # Prepare documents for coherence calculation
        tokenized_docs = []
        for doc in documents:
            tokens = doc.split()
            tokenized_docs.append(tokens)
        
        # Create word frequency dictionary
        word_freq = Counter()
        for doc_tokens in tokenized_docs:
            word_freq.update(doc_tokens)
        
        for num_topics in range(start, limit, step):
            print(f"   Training model with {num_topics} topics...")
            
            try:
                vectorizer = TfidfVectorizer(
                    max_features=1000,
                    min_df=2,
                    max_df=0.5,
                    token_pattern=r'\b\w+\b',
                    sublinear_tf=True,
                    smooth_idf=True
                )
                
                doc_term_matrix = vectorizer.fit_transform(documents)
                
                lda_model = LatentDirichletAllocation(
                    n_components=num_topics,
                    random_state=42,
                    max_iter=100,
                    learning_method='online',
                    learning_offset=50.0,
                    doc_topic_prior=0.1,
                    topic_word_prior=0.5,
                    batch_size=512,
                    evaluate_every=5,
                    perp_tol=1e-1,
                    mean_change_tol=1e-3
                )
                
                lda_model.fit(doc_term_matrix)
                
                # Calculate perplexity
                perplexity = lda_model.perplexity(doc_term_matrix)
                perplexity_values.append(perplexity)
                
                # Calculate coherence
                feature_names = vectorizer.get_feature_names_out()
                coherence = self.calculate_simplified_coherence(lda_model, feature_names, tokenized_docs, word_freq)
                coherence_values.append(coherence)
                
                model_list.append(lda_model)
                vectorizer_list.append(vectorizer)
                
                print(f"     Topics: {num_topics}, Perplexity: {perplexity:.4f}, Coherence: {coherence:.4f}")
                
            except Exception as e:
                print(f"     ❌ Training failed for {num_topics} topics: {e}")
                continue
        
        return model_list, vectorizer_list, coherence_values, perplexity_values
    
    def calculate_simplified_coherence(self, lda_model, feature_names, tokenized_docs, word_freq):
        """Calculate coherence score using Gensim if available, otherwise simplified version"""
        try:
            # Extract top words for each topic
            extracted_topics = []
            for topic_idx, topic in enumerate(lda_model.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                extracted_topics.append(top_words)
            
            if GENSIM_AVAILABLE:
                # Use standard c_v coherence with Gensim
                dictionary = Dictionary(tokenized_docs)
                dictionary.filter_extremes(no_below=2, no_above=0.5)
                
                coherence_model = CoherenceModel(
                    topics=extracted_topics,
                    texts=tokenized_docs,
                    dictionary=dictionary,
                    coherence='c_v'
                )
                coherence = coherence_model.get_coherence()
                return coherence
            else:
                # Fallback to simplified coherence calculation
                topic_coherences = []
                for topic_words in extracted_topics:
                    topic_coherence = self.calculate_topic_coherence(topic_words, tokenized_docs, word_freq)
                    topic_coherences.append(topic_coherence)
                
                return np.mean(topic_coherences) if topic_coherences else 0.0
            
        except Exception as e:
            print(f"     ⚠️ Coherence calculation failed: {e}")
            return 0.0
    
    def calculate_topic_coherence(self, topic_words, tokenized_docs, word_freq):
        """Calculate coherence for a single topic (fast version)"""
        try:
            if len(topic_words) < 2:
                return 0.0
            
            # Use only top 5 words for speed
            top_words = topic_words[:5]
            similarities = []
            
            # Simple co-occurrence calculation
            for i in range(len(top_words)):
                for j in range(i + 1, len(top_words)):
                    word1, word2 = top_words[i], top_words[j]
                    
                    # Use pre-calculated word frequencies
                    word1_count = word_freq.get(word1, 0)
                    word2_count = word_freq.get(word2, 0)
                    
                    # Quick co-occurrence count
                    co_occurrence = 0
                    for doc_tokens in tokenized_docs:
                        if word1 in doc_tokens and word2 in doc_tokens:
                            co_occurrence += 1
                    
                    # Calculate simple similarity
                    if co_occurrence > 0 and word1_count > 0 and word2_count > 0:
                        similarity = co_occurrence / min(word1_count, word2_count)
                        similarities.append(similarity)
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception as e:
            return 0.0
    
    def calculate_topic_diversity(self, lda_model, vectorizer, top_n=10):
        """
        Calculate topic diversity - how unique are the top words across topics
        Returns a value between 0 and 1, where 1 means perfect diversity (no overlap)
        """
        try:
            feature_names = vectorizer.get_feature_names_out()
            all_words = set()
            
            for topic in lda_model.components_:
                top_words = [feature_names[i] for i in topic.argsort()[-top_n:][::-1]]
                all_words.update(top_words)
            
            # Calculate diversity: unique words / (topics * words per topic)
            diversity = len(all_words) / (lda_model.n_components * top_n)
            return diversity
            
        except Exception as e:
            print(f"     ⚠️ Topic diversity calculation failed: {e}")
            return 0.0
    
    def get_unique_topic_words(self, lda_model, vectorizer, num_words=15, min_unique_words=5):
        """
        Get topic words with emphasis on unique words per topic
        Returns topics with unique words prioritized
        """
        try:
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            all_top_words = []
            
            # First pass: collect all top words
            for topic_idx, topic in enumerate(lda_model.components_):
                top_words_idx = topic.argsort()[-num_words:][::-1]
                top_words = [(feature_names[i], topic[i]) for i in top_words_idx]
                all_top_words.append(top_words)
            
            # Second pass: prioritize unique words
            used_words = set()
            for topic_idx, top_words in enumerate(all_top_words):
                unique_words = []
                common_words = []
                
                for word, weight in top_words:
                    if word not in used_words:
                        unique_words.append((word, weight))
                        used_words.add(word)
                    else:
                        common_words.append((word, weight))
                
                # Ensure minimum unique words, then add common words
                final_words = unique_words[:min_unique_words]
                remaining_slots = num_words - len(final_words)
                final_words.extend(common_words[:remaining_slots])
                
                # Sort by weight within the final selection
                final_words.sort(key=lambda x: x[1], reverse=True)
                topics.append([word for word, _ in final_words])
            
            return topics
            
        except Exception as e:
            print(f"     ⚠️ Unique topic words calculation failed: {e}")
            return []
    
    def plot_model_evaluation(self, start, limit, step, coherence_values, perplexity_values):
        """Plot model evaluation charts"""
        if not coherence_values or not perplexity_values:
            print("❌ No valid evaluation data")
            return -1
            
        x = list(range(start, start + len(coherence_values) * step, step))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Coherence plot
        ax1.plot(x, coherence_values, 'b-o')
        ax1.set_xlabel('Number of Topics')
        ax1.set_ylabel('Coherence Score (Higher is Better)')
        ax1.set_title('LDA Model Coherence')
        ax1.grid(True)
        best_coherence_idx = coherence_values.index(max(coherence_values))
        ax1.scatter(x[best_coherence_idx], coherence_values[best_coherence_idx], color='red', s=100, zorder=5)
        
        # Perplexity plot
        ax2.plot(x, perplexity_values, 'r-o')
        ax2.set_xlabel('Number of Topics')
        ax2.set_ylabel('Perplexity (Lower is Better)')
        ax2.set_title('LDA Model Perplexity')
        ax2.grid(True)
        best_perplexity_idx = perplexity_values.index(min(perplexity_values))
        ax2.scatter(x[best_perplexity_idx], perplexity_values[best_perplexity_idx], color='red', s=100, zorder=5)
        
        plt.tight_layout()
        plt.savefig('lda_hyperparameter_search.png', dpi=300, bbox_inches='tight')
        print("✅ Chart saved to lda_hyperparameter_search.png")
        plt.close()
        
        # Analysis
        print(f"\n📊 === Model Evaluation Analysis ===")
        best_perplexity_idx = perplexity_values.index(min(perplexity_values))
        best_coherence_idx = coherence_values.index(max(coherence_values))
        
        print(f"\n🔍 Metric interpretation:")
        print(f"   📈 Perplexity: Best {x[best_perplexity_idx]} topics, Perplexity: {min(perplexity_values):.2f}")
        print(f"   📊 Coherence: Best {x[best_coherence_idx]} topics, Coherence: {max(coherence_values):.4f}")
        
        # Recommendations
        print(f"\n🎯 === Model Selection Recommendations ===")
        if best_perplexity_idx == best_coherence_idx:
            print(f"✅ Strongly recommended: {x[best_perplexity_idx]} topics")
            recommended_idx = best_perplexity_idx
        else:
            print(f"🤔 Metrics have divergence:")
            print(f"   - Best perplexity: {x[best_perplexity_idx]} topics")
            print(f"   - Best coherence: {x[best_coherence_idx]} topics")
            print(f"✅ Recommended: {x[best_coherence_idx]} topics (coherence priority)")
            recommended_idx = best_coherence_idx
        
        # Results table
        print(f"\n📋 Complete results table:")
        print(f"{'Model':<4} {'Topics':<6} {'Perplexity':<12} {'Coherence':<12} {'Recommend':<4}")
        print("-" * 50)
        for i, (topics, coherence, perplexity) in enumerate(zip(x, coherence_values, perplexity_values)):
            marker = "👍" if i == recommended_idx else ""
            print(f"{i:<4} {topics:<6} {perplexity:<12.2f} {coherence:<12.4f} {marker:<4}")
        
        return recommended_idx
    
    def display_topics(self, lda_model, vectorizer, num_words=15, tokenized_docs=None, word_freq=None):
        """Display topic words with individual topic coherence and unique word prioritization"""
        feature_names = vectorizer.get_feature_names_out()
        topics = {}
        print("\n🏷️  === LDA Topic Analysis Results ===")
        
        # Calculate topic diversity
        diversity = self.calculate_topic_diversity(lda_model, vectorizer, top_n=10)
        print(f"📊 Topic Diversity Score: {diversity:.4f} (1.0 = perfect diversity)")
        
        # Get unique topic words
        # unique_topics = self.get_unique_topic_words(lda_model, vectorizer, num_words=num_words, min_unique_words=5)
        unique_topics = None  # Temporarily disable to test performance
        
        # Calculate individual topic coherence if tokenized_docs provided
        topic_coherences = []
        if tokenized_docs is not None and word_freq is not None:
            print("\n📊 Individual Topic Coherence Analysis:")
            for topic_idx, topic in enumerate(lda_model.components_):
                # Use unique topic words for coherence calculation
                if unique_topics and len(unique_topics) > topic_idx:
                    top_words = unique_topics[topic_idx][:10]  # Use top 10 for coherence
                else:
                    top_words_idx = topic.argsort()[-10:][::-1]
                    top_words = [feature_names[idx] for idx in top_words_idx]
                
                topic_coherence = self.calculate_topic_coherence(top_words, tokenized_docs, word_freq)
                topic_coherences.append(topic_coherence)
                print(f"   Topic {topic_idx + 1} Coherence: {topic_coherence:.4f}")
        
        for topic_idx, topic in enumerate(lda_model.components_):
            # Use unique topic words if available
            if unique_topics and len(unique_topics) > topic_idx:
                top_words = unique_topics[topic_idx]
                # Get weights for these words
                top_weights = []
                for word in top_words:
                    try:
                        word_idx = list(feature_names).index(word)
                        top_weights.append(topic[word_idx])
                    except ValueError:
                        top_weights.append(0.0)
            else:
                # Fallback to original method
                top_words_idx = topic.argsort()[-num_words:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                top_weights = [topic[i] for i in top_words_idx]
            
            topics[f"Topic {topic_idx + 1}"] = list(zip(top_words, top_weights))
            
            coherence_info = f" (Coherence: {topic_coherences[topic_idx]:.4f})" if topic_coherences else ""
            print(f"\n📌 Topic {topic_idx + 1}{coherence_info}:")
            for word, weight in zip(top_words[:10], top_weights[:10]):
                print(f"     {word}: {weight:.4f}")
        
        return topics, topic_coherences
    
    def calculate_topic_similarity_matrix(self, lda_model, vectorizer, alpha=0.01, beta=0.5, save_heatmap=True):
        """Calculate and visualize topic similarity matrix"""
        print("\n🔗 Calculating topic similarity matrix...")
        
        # Get topic-word distributions
        topic_word_distributions = lda_model.components_
        n_topics = lda_model.n_components
        
        # Calculate cosine similarity between topics
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(topic_word_distributions)
        
        # Create similarity matrix DataFrame
        topic_labels = [f'Topic {i+1}' for i in range(n_topics)]
        similarity_df = pd.DataFrame(similarity_matrix, 
                                   index=topic_labels, 
                                   columns=topic_labels)
        
        print("📊 Topic Similarity Matrix:")
        print(similarity_df.round(3))
        
        # Find most similar and least similar topic pairs
        print("\n🔍 Topic Similarity Analysis:")
        upper_triangle = np.triu(similarity_matrix, k=1)
        max_similarity_idx = np.unravel_index(np.argmax(upper_triangle), upper_triangle.shape)
        max_similarity = upper_triangle[max_similarity_idx]
        min_similarity_idx = np.unravel_index(np.argmin(upper_triangle), upper_triangle.shape)
        min_similarity = upper_triangle[min_similarity_idx]
        
        print(f"   🎯 Most similar topics: Topic {max_similarity_idx[0]+1} & Topic {max_similarity_idx[1]+1} (similarity: {max_similarity:.3f})")
        print(f"   📉 Least similar topics: Topic {min_similarity_idx[0]+1} & Topic {min_similarity_idx[1]+1} (similarity: {min_similarity:.3f})")
        
        avg_similarity = np.mean(upper_triangle)
        print(f"   📊 Average topic similarity: {avg_similarity:.3f}")
        
        # Save similarity matrix
        if save_heatmap:
            try:
                plt.figure(figsize=(10, 8))
                sns.heatmap(similarity_df, annot=True, cmap='RdYlBu_r', center=0.5, 
                           square=True, fmt='.3f', cbar_kws={'label': 'Cosine Similarity'})
                plt.title('Topic Similarity Matrix (Cosine Similarity)', fontsize=16, fontweight='bold')
                plt.tight_layout()
                
                heatmap_filename = f'topic_similarity_matrix_a{alpha}_b{beta}_t{lda_model.n_components}.png'
                plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight')
                print(f"✅ Topic similarity heatmap saved: {heatmap_filename}")
                plt.close()
                
                excel_filename = f'topic_similarity_matrix_a{alpha}_b{beta}_t{lda_model.n_components}.xlsx'
                similarity_df.to_excel(excel_filename)
                print(f"✅ Topic similarity matrix saved: {excel_filename}")
                
            except Exception as e:
                print(f"⚠️ Failed to save similarity matrix: {e}")
        
        return similarity_df, similarity_matrix
    
    def assign_topics_to_documents(self, lda_model, vectorizer, texts, df_processed):
        """Assign topic labels to each document"""
        print("\n🔖 Assigning topic labels to documents...")
        
        # Convert word segmentation results to documents
        documents = [' '.join(text) for text in texts]
        
        # Vectorization
        doc_term_matrix = vectorizer.transform(documents)
        
        # Get topic distribution
        doc_topic_probs = lda_model.transform(doc_term_matrix)
        
        # Get dominant topic
        dominant_topics = doc_topic_probs.argmax(axis=1)
        topic_probs = doc_topic_probs.max(axis=1)
        
        # Add to dataframe
        df_results = df_processed.copy()
        df_results['dominant_topic'] = dominant_topics
        df_results['topic_probability'] = topic_probs
        
        return df_results, dominant_topics, topic_probs
    
    def analyze_topic_distribution(self, df_results, alpha=0.01, beta=0.5, num_topics=None, create_heatmap=True):
        """Analyze topic distribution"""
        print("\n📊 Analyzing topic distribution...")
        
        dominant_topics = df_results['dominant_topic']
        
        # Topic distribution statistics
        topic_counts = pd.Series(dominant_topics).value_counts().sort_index()
        topic_percentages = (topic_counts / len(dominant_topics) * 100).round(2)
        
        print("📈 Document count and percentage for each topic:")
        for topic_id, count in topic_counts.items():
            percentage = topic_percentages[topic_id]
            print(f"   Topic {topic_id + 1}: {count} documents ({percentage}%)")
        
        # Create heatmap
        if create_heatmap and 'keyword_source' in df_results.columns:
            try:
                topic_by_keyword = pd.crosstab(
                    df_results['keyword_source'], 
                    df_results['dominant_topic']
                )
                
                plt.figure(figsize=(12, 8))
                sns.heatmap(topic_by_keyword, annot=True, fmt='d', cmap='YlOrRd')
                plt.title('Topic Distribution by Different Keywords', fontsize=16)
                plt.xlabel('Topic Number')
                plt.ylabel('Keyword Source')
                plt.tight_layout()
                
                if num_topics is None:
                    num_topics = len(df_results['dominant_topic'].unique())
                heatmap_filename = f'topic_distribution_heatmap_a{alpha}_b{beta}_t{num_topics}.png'
                plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight')
                print(f"✅ Topic distribution heatmap saved: {heatmap_filename}")
                plt.close()
            except Exception as e:
                print(f"⚠️ Heatmap generation failed: {e}")
        
        return topic_counts, topic_percentages
    
    def save_results(self, df_results, word_freq, topics, model_info, alpha=0.01, beta=0.5, num_topics=None, filename=None, topic_coherences=None):
        """Save analysis results with parameter-based naming"""
        if filename is None:
            if num_topics is None:
                num_topics = model_info.get('Number of Topics', 'unknown')
            filename = f'optimized_lda_results_a{alpha}_b{beta}_t{num_topics}.xlsx'
        
        print(f"\n💾 Saving analysis results to {filename}...")
        
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Save document topic assignments
                if df_results is not None:
                    df_results.to_excel(writer, sheet_name='Document Topic Assignment', index=False)
                
                # Save word frequency statistics
                word_freq_df = pd.DataFrame([
                    {'Word': word, 'Frequency': freq} 
                    for word, freq in word_freq.most_common(100)
                ])
                word_freq_df.to_excel(writer, sheet_name='Word Frequency Statistics', index=False)
                
                # Save topic words with coherence
                if topics:
                    topic_words_data = []
                    for i, (topic_name, words_weights) in enumerate(topics.items()):
                        coherence = topic_coherences[i] if topic_coherences and i < len(topic_coherences) else None
                        for word, weight in words_weights:
                            topic_words_data.append({
                                'Topic': topic_name,
                                'Word': word,
                                'Weight': weight,
                                'Topic Coherence': coherence
                            })
                    
                    pd.DataFrame(topic_words_data).to_excel(writer, sheet_name='Topic Words', index=False)
                
                # Save topic distribution statistics with coherence
                if df_results is not None and 'dominant_topic' in df_results.columns:
                    topic_stats = df_results['dominant_topic'].value_counts().sort_index()
                    topic_stats_data = []
                    for i, (topic_idx, count) in enumerate(topic_stats.items()):
                        coherence = topic_coherences[topic_idx] if topic_coherences and topic_idx < len(topic_coherences) else None
                        topic_stats_data.append({
                            'Topic Number': topic_idx + 1,
                            'Document Count': count,
                            'Percentage(%)': round((count / len(df_results) * 100), 2),
                            'Topic Coherence': coherence
                        })
                    topic_stats_df = pd.DataFrame(topic_stats_data)
                    topic_stats_df.to_excel(writer, sheet_name='Topic Distribution Statistics', index=False)
                
                # Save model information with complete parameters
                if model_info:
                    # Ensure all model parameters are included
                    complete_model_info = {
                        'Number of Topics': model_info.get('Number of Topics', 'N/A'),
                        'Alpha (doc_topic_prior)': model_info.get('Alpha', model_info.get('Best Alpha', 0.1)),
                        'Beta (topic_word_prior)': model_info.get('Beta', model_info.get('Best Beta', 0.5)),
                        'Perplexity': model_info.get('Perplexity', 'N/A'),
                        'Coherence (c_v)': model_info.get('Coherence', model_info.get('Evaluation Score', 'N/A')),
                        'Topic Diversity': model_info.get('Topic Diversity', 'N/A'),
                        'Mode': model_info.get('Mode', 'N/A'),
                        'Batch Size': 512,
                        'Max Iterations': 100,
                        'Learning Method': 'online',
                        'Learning Offset': 50.0,
                        'Random State': 42
                    }
                    
                    # Add search range if available
                    if 'Search Range' in model_info:
                        complete_model_info['Search Range'] = model_info['Search Range']
                    if 'Selected Index' in model_info:
                        complete_model_info['Selected Index'] = model_info['Selected Index']
                    
                    model_info_df = pd.DataFrame([complete_model_info])
                    model_info_df.to_excel(writer, sheet_name='Model Information', index=False)
                
                # Save topic coherence summary
                if topic_coherences:
                    coherence_summary = []
                    for i, coherence in enumerate(topic_coherences):
                        coherence_summary.append({
                            'Topic Number': i + 1,
                            'Coherence Score': coherence,
                            'Interpretation': 'High' if coherence > 0.6 else 'Medium' if coherence > 0.3 else 'Low'
                        })
                    coherence_df = pd.DataFrame(coherence_summary)
                    coherence_df.to_excel(writer, sheet_name='Topic Coherence Summary', index=False)
            
            print("✅ Results saved successfully")
        except Exception as e:
            print(f"❌ Save failed: {e}")

def main():
    # Initialize analyzer
    analyzer = OptimizedLDAAnalyzer()
    
    print("🔍 Xiaohongshu Feminism Topic LDA Analysis (Optimized Version)")
    print("=" * 60)
    
    # Get data directory
    print("📁 File detection options:")
    print("   1. Current directory")
    print("   2. Specify directory")
    print("   3. MediaCrawler data directory")
    
    choice = input("Please choose (1-3, default 1): ").strip()
    
    if choice == "2":
        directory = input("Please enter CSV file directory: ").strip() or "."
    elif choice == "3":
        directory = "/Users/imrayne/MediaCrawler/data/xhs"
        if not os.path.exists(directory):
            print(f"❌ Directory does not exist: {directory}")
            directory = "."
    else:
        directory = "."
    
    # Load data
    df = analyzer.load_data(directory=directory)
    if df.empty:
        print("❌ No data to analyze")
        return
    
    # Data preprocessing
    texts, df_processed = analyzer.preprocess_data(df)
    if not texts:
        print("❌ No valid data after preprocessing")
        return
    
    # Word frequency analysis
    word_freq, top_words = analyzer.create_word_frequency_analysis(texts)
    
    # Convert word segmentation results to document format
    documents = [' '.join(text) for text in texts]
    
    # Topic modeling settings
    print("\n⚙️ Topic modeling settings:")
    print("   1. Automatic hyperparameter search for optimal number of topics")
    print("   2. Manual specification of number of topics")
    print("   3. Quick mode (recommended number of topics)")
    print("   4. Grid search for optimal alpha and beta parameters")
    print("   5. Custom parameters (specify alpha, beta, and topic number)")
    
    mode = input("Please choose mode (1-5, default 3): ").strip()
    if not mode:
        mode = "3"
    
    if mode == "4":
        # Grid search for alpha and beta
        num_topics = int(input("Please enter number of topics (default 8): ").strip() or "8")
        
        print(f"\n🔍 Starting grid search for optimal alpha and beta parameters...")
        best_model, best_vectorizer, best_alpha, best_beta, grid_results = analyzer.grid_search_alpha_beta(
            documents, num_topics
        )
        
        if best_model is None:
            print("❌ Grid search failed")
            return
        
        # Set the variables for the rest of the analysis
        lda_model = best_model
        vectorizer = best_vectorizer
        
        # Calculate coherence for best model
        tokenized_docs = []
        for doc in documents:
            tokens = doc.split()
            tokenized_docs.append(tokens)
        
        word_freq = Counter()
        for doc_tokens in tokenized_docs:
            word_freq.update(doc_tokens)
        
        feature_names = best_vectorizer.get_feature_names_out()
        coherence = analyzer.calculate_simplified_coherence(best_model, feature_names, tokenized_docs, word_freq)
        
        model_info = {
            'Number of Topics': num_topics,
            'Best Alpha': best_alpha,
            'Best Beta': best_beta,
            'Perplexity': best_model.perplexity(best_vectorizer.transform(documents)),
            'Coherence': coherence,
            'Mode': 'Grid Search Alpha-Beta'
        }
        
        print(f"✅ Grid search completed, using best model with alpha={best_alpha}, beta={best_beta}")
        
    elif mode == "2":
        num_topics = int(input("Please enter number of topics (default 8): ").strip() or "8")
        
        # Directly train model with specified number of topics
        print(f"\n🤖 Training LDA model with {num_topics} topics...")
        
        vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.5,
            token_pattern=r'\b\w+\b',
            sublinear_tf=True,
            smooth_idf=True
        )
        
        doc_term_matrix = vectorizer.fit_transform(documents)
        
        lda_model = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            max_iter=100,
            learning_method='online',
            learning_offset=50.0,
            doc_topic_prior=0.01,
            topic_word_prior=0.5,
            batch_size=128,
            evaluate_every=5,
            perp_tol=1e-1,
            mean_change_tol=1e-3
        )
        
        lda_model.fit(doc_term_matrix)
        
        # Calculate coherence
        tokenized_docs = []
        for doc in documents:
            tokens = doc.split()
            tokenized_docs.append(tokens)
        
        word_freq = Counter()
        for doc_tokens in tokenized_docs:
            word_freq.update(doc_tokens)
        
        feature_names = vectorizer.get_feature_names_out()
        coherence = analyzer.calculate_simplified_coherence(lda_model, feature_names, tokenized_docs, word_freq)
        
        model_info = {
            'Number of Topics': num_topics,
            'Alpha': 0.1,
            'Beta': 0.5,
            'Perplexity': lda_model.perplexity(doc_term_matrix),
            'Coherence': coherence,
            'Mode': 'Manual Specification'
        }
        
    elif mode == "3":
        # Quick mode: use recommended number of topics
        print(f"\n⚡ Quick mode: Recommend number of topics based on data scale")
        
        # Recommend number of topics based on document count
        doc_count = len(documents)
        if doc_count < 500:
            recommended_topics = 3
        elif doc_count < 1000:
            recommended_topics = 5
        elif doc_count < 2000:
            recommended_topics = 8
        else:
            recommended_topics = 10
            
        print(f"📊 Document count: {doc_count}")
        print(f"🎯 Recommended number of topics: {recommended_topics}")
        
        confirm = input(f"Use recommended {recommended_topics} topics? (y/n, default y): ").strip().lower()
        if confirm and confirm.startswith('n'):
            recommended_topics = int(input("Please enter number of topics: ").strip())
        
        print(f"\n🤖 Training LDA model with {recommended_topics} topics...")
        
        vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.5,
            token_pattern=r'\b\w+\b',
            sublinear_tf=True,
            smooth_idf=True
        )
        
        doc_term_matrix = vectorizer.fit_transform(documents)
        
        lda_model = LatentDirichletAllocation(
            n_components=recommended_topics,
            random_state=42,
            max_iter=100,
            learning_method='online',
            learning_offset=50.0,
            doc_topic_prior=0.1,
            topic_word_prior=0.5,
            batch_size=512,
            evaluate_every=5,
            perp_tol=1e-1,
            mean_change_tol=1e-3
        )
        
        lda_model.fit(doc_term_matrix)
        
        # Calculate coherence
        tokenized_docs = []
        for doc in documents:
            tokens = doc.split()
            tokenized_docs.append(tokens)
        
        word_freq = Counter()
        for doc_tokens in tokenized_docs:
            word_freq.update(doc_tokens)
        
        feature_names = vectorizer.get_feature_names_out()
        coherence = analyzer.calculate_simplified_coherence(lda_model, feature_names, tokenized_docs, word_freq)
        
        model_info = {
            'Number of Topics': recommended_topics,
            'Alpha': 0.1,
            'Beta': 0.5,
            'Perplexity': lda_model.perplexity(doc_term_matrix),
            'Coherence': coherence,
            'Mode': 'Quick Mode'
        }
        
        print(f"✅ Model training completed, perplexity: {model_info['Perplexity']:.2f}")
        
    elif mode == "5":
        # Custom parameters mode
        print(f"\n🔧 Custom parameters mode")
        
        # Get custom parameters
        custom_alpha = float(input("Please enter alpha value (default 0.1): ").strip() or "0.1")
        custom_beta = float(input("Please enter beta value (default 0.5): ").strip() or "0.5")
        custom_topics = int(input("Please enter number of topics (default 8): ").strip() or "8")
        
        print(f"\n🤖 Training LDA model with custom parameters:")
        print(f"   Alpha: {custom_alpha}")
        print(f"   Beta: {custom_beta}")
        print(f"   Topics: {custom_topics}")
        
        vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.5,
            token_pattern=r'\b\w+\b',
            sublinear_tf=True,
            smooth_idf=True
        )
        
        doc_term_matrix = vectorizer.fit_transform(documents)
        
        lda_model = LatentDirichletAllocation(
            n_components=custom_topics,
            random_state=42,
            max_iter=100,
            learning_method='online',
            learning_offset=50.0,
            doc_topic_prior=custom_alpha,
            topic_word_prior=custom_beta,
            batch_size=512,
            evaluate_every=5,
            perp_tol=1e-1,
            mean_change_tol=1e-3
        )
        
        lda_model.fit(doc_term_matrix)
        
        # Calculate coherence
        tokenized_docs = []
        for doc in documents:
            tokens = doc.split()
            tokenized_docs.append(tokens)
        
        word_freq = Counter()
        for doc_tokens in tokenized_docs:
            word_freq.update(doc_tokens)
        
        feature_names = vectorizer.get_feature_names_out()
        coherence = analyzer.calculate_simplified_coherence(lda_model, feature_names, tokenized_docs, word_freq)
        
        model_info = {
            'Number of Topics': custom_topics,
            'Alpha': custom_alpha,
            'Beta': custom_beta,
            'Perplexity': lda_model.perplexity(doc_term_matrix),
            'Coherence': coherence,
            'Mode': 'Custom Parameters'
        }
        
        print(f"✅ Model training completed")
        print(f"   Perplexity: {model_info['Perplexity']:.2f}")
        print(f"   Coherence: {model_info['Coherence']:.4f}")
        
    else:
        # Hyperparameter search
        start_range = int(input("Starting number of topics (default 2): ").strip() or "2")
        end_range = int(input("Ending number of topics (default 16): ").strip() or "16")
        step_size = int(input("Step size (default 1): ").strip() or "1")
        
        model_list, vectorizer_list, coherence_values, perplexity_values = analyzer.compute_coherence_values(
            documents, start=start_range, limit=end_range, step=step_size
        )
        
        if not model_list:
            print("❌ Hyperparameter search failed")
            return
        
        print("🎨 Starting to plot evaluation charts...")
        best_idx = analyzer.plot_model_evaluation(start_range, end_range, step_size, coherence_values, perplexity_values)
        
        if best_idx == -1:
            print("❌ Cannot determine best model")
            return
            
        print("📊 Evaluation chart plotting completed")
        
        # Let user choose model
        print(f"\n🎯 Recommended to use model {best_idx} (index starts from 0)")
        print(f"🔧 Debug info: Total of {len(model_list)} models available")
        
        try:
            model_choice = input(f"Please choose model index (0-{len(model_list)-1}, default {best_idx}): ").strip()
            model_idx = int(model_choice) if model_choice.isdigit() and 0 <= int(model_choice) < len(model_list) else best_idx
            print(f"✅ User chose model {model_idx}")
        except Exception as e:
            print(f"⚠️ User input exception: {e}, using recommended model {best_idx}")
            model_idx = best_idx
        
        lda_model = model_list[model_idx]
        vectorizer = vectorizer_list[model_idx]
        
        model_info = {
            'Number of Topics': lda_model.n_components,
            'Alpha': 0.1,
            'Beta': 0.5,
            'Perplexity': perplexity_values[model_idx],
            'Coherence': coherence_values[model_idx],
            'Mode': 'Hyperparameter Search',
            'Search Range': f'{start_range}-{end_range-1}',
            'Selected Index': model_idx
        }
        
        print(f"✅ Selected model with {lda_model.n_components} topics")
    
    # Display topics
    print("📝 Starting to display topic words...")
    
    # Prepare tokenized documents for coherence calculation
    tokenized_docs = []
    for doc in documents:
        tokens = doc.split()
        tokenized_docs.append(tokens)
    
    # Create word frequency dictionary for coherence calculation
    word_freq = Counter()
    for doc_tokens in tokenized_docs:
        word_freq.update(doc_tokens)
    
    topics, topic_coherences = analyzer.display_topics(lda_model, vectorizer, tokenized_docs=tokenized_docs, word_freq=word_freq)
    print("✅ Topic word display completed")
    
    # Calculate topic similarity matrix
    similarity_df, similarity_matrix = analyzer.calculate_topic_similarity_matrix(lda_model, vectorizer)
    
    # Assign topics to documents
    print("🔖 Starting to assign topics to documents...")
    df_results, dominant_topics, topic_probs = analyzer.assign_topics_to_documents(
        lda_model, vectorizer, texts, df_processed
    )
    print("✅ Document topic assignment completed")
    
    # Analyze topic distribution
    print("📊 Starting to analyze topic distribution...")
    topic_counts, topic_percentages = analyzer.analyze_topic_distribution(df_results)
    print("✅ Topic distribution analysis completed")
    
    # Save results
    analyzer.save_results(df_results, word_freq, topics, model_info, topic_coherences=topic_coherences)
    
    print("\n🎉 === Analysis Completed ===")
    print("📁 Generated files:")
    print("   • optimized_lda_results.xlsx - Detailed analysis results")
    print("   • lda_hyperparameter_search.png - Hyperparameter search result charts")
    print("   • topic_distribution_heatmap.png - Topic distribution heatmap")

if __name__ == "__main__":
    main() 