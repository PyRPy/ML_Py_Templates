# -- Latent Dirichlet allocation ---

# In Chapter 5 we briefly introduced the AssociatedPress dataset provided by the topicmodels package, 
# as an example of a DocumentTermMatrix. This is a collection of 2246 news articles from an American news 
# agency, mostly published around 1988.

library(topicmodels)

data("AssociatedPress")
AssociatedPress

# set a seed so that the output of the model is predictable
# k = 2 , two topics
ap_lda <- LDA(AssociatedPress, k = 2, control = list(seed = 1234))
ap_lda

# Word-topic probabilities
# tidytext package provides method for extracting the per-topic-per-word probabilities, called ??
# ("beta"), from the model.
library(tidytext)

ap_topics <- tidy(ap_lda, matrix = "beta")
ap_topics

# use dplyr's top_n() to find the 10 terms that are most common within each topic
library(ggplot2)
library(dplyr)

ap_top_terms <- ap_topics %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

ap_top_terms %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip()

# consider the terms that had the greatest difference in  ??between topic 1 and topic 2. This can be 
# estimated based on the log ratio of the two
library(tidyr)

beta_spread <- ap_topics %>%
  mutate(topic = paste0("topic", topic)) %>%
  spread(topic, beta) %>%
  filter(topic1 > .001 | topic2 > .001) %>%
  mutate(log_ratio = log2(topic2 / topic1))

beta_spread

# Document-topic probabilities
# Besides estimating each topic as a mixture of words, LDA also models each document as a mixture of topics.
ap_documents <- tidy(ap_lda, matrix = "gamma")
ap_documents

# document 6 was drawn almost entirely from topic 2, having a  ?? from topic 1 close to zero. 
tidy(AssociatedPress) %>%
  filter(document == 6) %>%
  arrange(desc(count))

# --- Example: the great library heist ---
# This vandal has torn the books into individual chapters, and left them in one large pile. How can 
# we restore these disorganized chapters to their original books? 
titles <- c("Twenty Thousand Leagues under the Sea", "The War of the Worlds",
            "Pride and Prejudice", "Great Expectations")

library(gutenbergr)

books <- gutenberg_works(title %in% titles) %>%
  gutenberg_download(meta_fields = "title")

# divide these into chapters, use tidytext's unnest_tokens() to separate them into words, then remove stop_words
library(stringr)

# divide into documents, each representing one chapter
by_chapter <- books %>%
  group_by(title) %>%
  mutate(chapter = cumsum(str_detect(text, regex("^chapter ", ignore_case = TRUE)))) %>%
  ungroup() %>%
  filter(chapter > 0) %>%
  unite(document, title, chapter)

# split into words
by_chapter_word <- by_chapter %>%
  unnest_tokens(word, text)

# find document-word counts
word_counts <- by_chapter_word %>%
  anti_join(stop_words) %>%
  count(document, word, sort = TRUE) %>%
  ungroup()

word_counts

# LDA on chapters
# word_counts is in a tidy form, with one-term-per-document-per-row, but the topicmodels package 
# requires a DocumentTermMatrix
chapters_dtm <- word_counts %>%
  cast_dtm(document, word, n)

chapters_dtm

#  use the LDA() function to create a four-topic model. In this case we know we're looking for four 
# topics because there are four books
chapters_lda <- LDA(chapters_dtm, k = 4, control = list(seed = 1234))
chapters_lda

chapter_topics <- tidy(chapters_lda, matrix = "beta")
chapter_topics

#  the term "joe" has an almost zero probability of being generated from topics 1, 2, or 3, 
# but it makes up 1.45% of topic 4

top_terms <- chapter_topics %>%
  group_by(topic) %>%
  top_n(5, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

top_terms

library(ggplot2)

top_terms %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip()

# Per-document classification
# Can we put the chapters back together in the correct books? We can find this by examining 
# the per-document-per-topic probabilities,  ??("gamma").

chapters_gamma <- tidy(chapters_lda, matrix = "gamma")
chapters_gamma

# Now that we have these topic probabilities, we can see how well our unsupervised learning 
# did at distinguishing the four books.
chapters_gamma <- chapters_gamma %>%
  separate(document, c("title", "chapter"), sep = "_", convert = TRUE)

chapters_gamma

# reorder titles in order of topic 1, topic 2, etc before plotting
chapters_gamma %>%
  mutate(title = reorder(title, gamma * topic)) %>%
  ggplot(aes(factor(topic), gamma)) +
  geom_boxplot() +
  facet_wrap(~ title)

# We notice that almost all of the chapters from Pride and Prejudice, War of the Worlds, and 
# Twenty Thousand Leagues Under the Sea were uniquely identified as a single topic each.

# some chapters from Great Expectations (which should be topic 4) were somewhat associated 
# with other topics.
chapter_classifications <- chapters_gamma %>%
  group_by(title, chapter) %>%
  top_n(1, gamma) %>%
  ungroup()

chapter_classifications

# compare each to the "consensus" topic for each book (the most common topic among its chapters
book_topics <- chapter_classifications %>%
  count(title, topic) %>%
  group_by(title) %>%
  top_n(1, n) %>%
  ungroup() %>%
  transmute(consensus = title, topic)

chapter_classifications %>%
  inner_join(book_topics, by = "topic") %>%
  filter(title != consensus)

# By word assignments: augment
# take the original document-word pairs and find which words in each document were 
# assigned to which topic.
assignments <- augment(chapters_lda, data = chapters_dtm)
assignments

# combine this assignments table with the consensus book titles to find which words 
# were incorrectly classified.
assignments <- assignments %>%
  separate(document, c("title", "chapter"), sep = "_", convert = TRUE) %>%
  inner_join(book_topics, by = c(".topic" = "topic"))

assignments

# visualize a confusion matrix, showing how often words from one book were assigned to 
# another, using dplyr's count() and ggplot2's geom_tile
assignments %>%
  count(title, consensus, wt = count) %>%
  group_by(title) %>%
  mutate(percent = n / sum(n)) %>%
  ggplot(aes(consensus, title, fill = percent)) +
  geom_tile() +
  scale_fill_gradient2(high = "red") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1),
        panel.grid = element_blank()) +
  labs(x = "Book words were assigned to",
       y = "Book words came from",
       fill = "% of assignments")

# We notice that almost all the words for Pride and Prejudice, Twenty Thousand Leagues 
# Under the Sea, and War of the Worlds were correctly assigned, while Great Expectations 
# had a fair number of misassigned words

# What were the most commonly mistaken words?
wrong_words <- assignments %>%
  filter(title != consensus)

wrong_words

wrong_words %>%
  count(title, consensus, term, wt = count) %>%
  ungroup() %>%
  arrange(desc(n))
# a number of words were often assigned to the Pride and Prejudice or War of the Worlds 
# cluster even when they appeared in Great Expectations. 

word_counts %>%
  filter(word == "flopson")

# --- Alternative LDA implementations ---
# mallet package : takes non-tokenized documents and performs the tokenization itself, 
# and requires a separate file of stopwords
library(mallet)

# create a vector with one string per chapter
collapsed <- by_chapter_word %>%
  anti_join(stop_words, by = "word") %>%
  mutate(word = str_replace(word, "'", "")) %>%
  group_by(document) %>%
  summarize(text = paste(word, collapse = " "))

# create an empty file of "stopwords"
file.create(empty_file <- tempfile())
docs <- mallet.import(collapsed$document, collapsed$text, empty_file)

mallet_model <- MalletLDA(num.topics = 4)
mallet_model$loadDocuments(docs)
mallet_model$train(100)

# Once the model is created, however, we can use the tidy() and augment() functions 
# described in the rest of the chapter in an almost identical way. 
# word-topic pairs
tidy(mallet_model)

# document-topic pairs
tidy(mallet_model, matrix = "gamma")

# column needs to be named "term" for "augment"
term_counts <- rename(word_counts, term = word)
augment(mallet_model, term_counts)
