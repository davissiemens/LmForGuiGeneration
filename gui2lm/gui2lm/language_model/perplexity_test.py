# import tensorflow as tf
#
# from gui2lm.gui2lm.language_model.languagemodel_with_hypertuning import perplexity
#
# if __name__ == '__main__':
#     tf.random.set_seed(42)
#     target = tf.random.uniform(shape=[2, 5], maxval=10, dtype=tf.int32, seed=42)
#     logits = tf.random.uniform(shape=(2, 5, 10), seed=42)
#
#     print("target")
#     tf.print(target)
#     print("logits")
#     tf.print(logits)
#
#     perplexity = perplexity(target, logits)
#     print("perplexity", perplexity)
