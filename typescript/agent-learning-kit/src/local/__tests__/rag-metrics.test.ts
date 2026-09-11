/**
 * Tests for RAG metrics
 */

import {
    contextPrecision,
    contextRecall,
    faithfulness,
    groundedness,
    answerRelevance,
    contextRelevance,
    contextUtilization,
    tokenize,
    jaccardSimilarity,
    ngramOverlap,
    normalizeContext,
    extractSentences,
    RAGInput
} from '../metrics/rag';

describe('RAG Utility Functions', () => {
    describe('tokenize', () => {
        it('should tokenize text into words', () => {
            const tokens = tokenize('Hello, World! How are you?');
            expect(tokens).toEqual(['hello', 'world', 'how', 'are', 'you']);
        });

        it('should handle empty text', () => {
            expect(tokenize('')).toEqual([]);
        });

        it('should lowercase tokens', () => {
            expect(tokenize('HELLO World')).toEqual(['hello', 'world']);
        });
    });

    describe('jaccardSimilarity', () => {
        it('should return 1 for identical sets', () => {
            const tokens = ['a', 'b', 'c'];
            expect(jaccardSimilarity(tokens, tokens)).toBe(1);
        });

        it('should return 0 for disjoint sets', () => {
            expect(jaccardSimilarity(['a', 'b'], ['c', 'd'])).toBe(0);
        });

        it('should calculate correct similarity', () => {
            const sim = jaccardSimilarity(['a', 'b', 'c'], ['b', 'c', 'd']);
            expect(sim).toBeCloseTo(0.5, 2); // 2 common / 4 total
        });
    });

    describe('ngramOverlap', () => {
        it('should calculate n-gram overlap', () => {
            const overlap = ngramOverlap('machine learning is great', 'machine learning works', 2);
            expect(overlap).toBeGreaterThan(0);
        });

        it('should return 0 for no overlap', () => {
            const overlap = ngramOverlap('hello world', 'foo bar', 2);
            expect(overlap).toBe(0);
        });
    });

    describe('normalizeContext', () => {
        it('should pass through arrays', () => {
            const contexts = ['a', 'b', 'c'];
            expect(normalizeContext(contexts)).toEqual(contexts);
        });

        it('should split string by paragraphs', () => {
            const context = 'First paragraph.\n\nSecond paragraph.';
            const normalized = normalizeContext(context);
            expect(normalized.length).toBe(2);
        });

        it('should handle single string', () => {
            const context = 'Single context chunk';
            expect(normalizeContext(context)).toEqual([context]);
        });
    });

    describe('extractSentences', () => {
        it('should split text into sentences', () => {
            const sentences = extractSentences('First sentence. Second sentence! Third?');
            expect(sentences.length).toBe(3);
        });

        it('should handle empty text', () => {
            expect(extractSentences('')).toEqual([]);
        });
    });
});

describe('Context Precision', () => {
    it('should calculate precision with ground truth contexts', () => {
        const input: RAGInput = {
            query: 'What is machine learning?',
            response: 'Machine learning is a type of AI.',
            context: [
                'Machine learning is a branch of artificial intelligence.',
                'Recipe for chocolate cake with eggs and flour.'
            ],
            ground_truth_contexts: ['Machine learning is a branch of artificial intelligence.']
        };

        const result = contextPrecision(input);
        expect(result.score).toBe(0.5); // 1 of 2 relevant
        expect(result.relevantIndices).toEqual([0]);
    });

    it('should use query relevance when no ground truth', () => {
        const input: RAGInput = {
            query: 'What is machine learning?',
            response: 'ML is AI.',
            context: [
                'Machine learning uses data to learn patterns.',
                'The weather is nice today.'
            ]
        };

        const result = contextPrecision(input);
        // With word-based similarity, at least the first context should be somewhat relevant
        expect(result.chunkScores).toHaveLength(2);
        // First context should have higher score than second
        expect(result.chunkScores![0]).toBeGreaterThan(result.chunkScores![1]);
    });

    it('should return 0 for no context', () => {
        const input: RAGInput = {
            query: 'What is AI?',
            response: 'AI is artificial intelligence.',
            context: []
        };

        const result = contextPrecision(input);
        expect(result.score).toBe(0);
        expect(result.passed).toBe(false);
    });
});

describe('Context Recall', () => {
    it('should calculate recall with ground truth contexts', () => {
        const input: RAGInput = {
            query: 'What is machine learning?',
            response: 'ML is a type of AI.',
            context: ['Machine learning is a branch of AI.'],
            ground_truth_contexts: [
                'Machine learning is a branch of AI.',
                'ML uses statistical methods.'
            ]
        };

        const result = contextRecall(input);
        expect(result.score).toBe(0.5); // Found 1 of 2 ground truths
        expect(result.recall).toBe(0.5);
    });

    it('should use expected response when no ground truth', () => {
        const input: RAGInput = {
            query: 'What is AI?',
            response: 'AI is artificial intelligence.',
            context: ['Artificial intelligence simulates human thinking.'],
            expected_response: 'AI is artificial intelligence that mimics human cognition.'
        };

        const result = contextRecall(input);
        expect(result.score).toBeGreaterThan(0);
    });

    it('should return full recall when no references', () => {
        const input: RAGInput = {
            query: 'What is AI?',
            response: 'AI is artificial intelligence.',
            context: ['Some context.']
        };

        const result = contextRecall(input);
        expect(result.score).toBe(1.0);
        expect(result.passed).toBe(true);
    });
});

describe('Faithfulness', () => {
    it('should score high for faithful responses', () => {
        const input: RAGInput = {
            query: 'What is the capital of France?',
            response: 'Paris is the capital of France.',
            context: ['Paris is the capital city of France and its largest city.']
        };

        const result = faithfulness(input);
        expect(result.score).toBeGreaterThan(0.5);
        expect(result.passed).toBe(true);
    });

    it('should score low for unfaithful responses', () => {
        const input: RAGInput = {
            query: 'What is the capital of France?',
            response: 'London is the capital of France. It has the Big Ben.',
            context: ['Paris is the capital city of France.']
        };

        const result = faithfulness(input);
        expect(result.score).toBeLessThan(0.8);
    });

    it('should return 0 for no response', () => {
        const input: RAGInput = {
            query: 'What is AI?',
            response: '',
            context: ['AI is artificial intelligence.']
        };

        const result = faithfulness(input);
        expect(result.score).toBe(0);
        expect(result.passed).toBe(false);
    });

    it('should return 0 for no context', () => {
        const input: RAGInput = {
            query: 'What is AI?',
            response: 'AI is artificial intelligence.',
            context: []
        };

        const result = faithfulness(input);
        expect(result.score).toBe(0);
        expect(result.passed).toBe(false);
    });

    it('should extract claims from response', () => {
        const input: RAGInput = {
            query: 'What is AI?',
            response: 'AI is artificial intelligence. It can learn from data.',
            context: ['Artificial intelligence learns from data and makes predictions.']
        };

        const result = faithfulness(input);
        expect(result.claims?.length).toBe(2);
    });
});

describe('Groundedness (alias for Faithfulness)', () => {
    it('should work the same as faithfulness', () => {
        const input: RAGInput = {
            query: 'What is AI?',
            response: 'AI is artificial intelligence.',
            context: ['Artificial intelligence is a field of computer science.']
        };

        const faithResult = faithfulness(input);
        const groundResult = groundedness(input);

        expect(groundResult.score).toBe(faithResult.score);
    });
});

describe('Answer Relevance', () => {
    it('should score high for relevant responses', () => {
        const input: RAGInput = {
            query: 'What is machine learning?',
            response: 'Machine learning is a subset of artificial intelligence that enables systems to learn from data.',
            context: ['ML context...']
        };

        const result = answerRelevance(input);
        // Response contains key terms from query like "machine", "learning"
        expect(result.score).toBeGreaterThan(0.2);
        // With 0.5 threshold, check if it passes
        expect(result.chunkScores).toHaveLength(3); // token, ngram, keyterm scores
    });

    it('should score low for irrelevant responses', () => {
        const input: RAGInput = {
            query: 'What is the capital of France?',
            response: 'Chocolate cake is delicious and easy to make at home.',
            context: ['Context about France...']
        };

        const result = answerRelevance(input);
        expect(result.score).toBeLessThan(0.5);
    });

    it('should return 0 for no response', () => {
        const input: RAGInput = {
            query: 'What is AI?',
            response: '',
            context: ['Context...']
        };

        const result = answerRelevance(input);
        expect(result.score).toBe(0);
        expect(result.passed).toBe(false);
    });

    it('should return 1 for no query', () => {
        const input: RAGInput = {
            query: '',
            response: 'Some response.',
            context: ['Context...']
        };

        const result = answerRelevance(input);
        expect(result.score).toBe(1.0);
    });
});

describe('Context Relevance', () => {
    it('should score high for relevant context', () => {
        const input: RAGInput = {
            query: 'What is machine learning?',
            response: 'ML is AI.',
            context: [
                'Machine learning is a branch of artificial intelligence that uses data.',
                'Deep learning is a subset of machine learning.'
            ]
        };

        const result = contextRelevance(input);
        expect(result.score).toBeGreaterThan(0.3);
    });

    it('should score low for irrelevant context', () => {
        const input: RAGInput = {
            query: 'What is the capital of France?',
            response: 'Paris.',
            context: [
                'Recipe for chocolate cake.',
                'Weather forecast for tomorrow.'
            ]
        };

        const result = contextRelevance(input);
        expect(result.score).toBeLessThan(0.5);
    });

    it('should return 0 for no context', () => {
        const input: RAGInput = {
            query: 'What is AI?',
            response: 'AI is...',
            context: []
        };

        const result = contextRelevance(input);
        expect(result.score).toBe(0);
    });
});

describe('Context Utilization', () => {
    it('should score high when context is utilized', () => {
        const input: RAGInput = {
            query: 'What is machine learning?',
            response: 'Machine learning is a branch of artificial intelligence that uses algorithms to learn from data.',
            context: [
                'Machine learning is a branch of artificial intelligence.',
                'ML algorithms learn from data and improve over time.'
            ]
        };

        const result = contextUtilization(input);
        expect(result.score).toBeGreaterThan(0.3);
    });

    it('should score low when context is not utilized', () => {
        const input: RAGInput = {
            query: 'What is machine learning?',
            response: 'Quantum physics studies subatomic particles.',
            context: [
                'Machine learning uses data to learn.',
                'AI algorithms process information.'
            ]
        };

        const result = contextUtilization(input);
        expect(result.score).toBeLessThan(0.5);
    });

    it('should return 0 for no context', () => {
        const input: RAGInput = {
            query: 'What is AI?',
            response: 'AI is artificial intelligence.',
            context: []
        };

        const result = contextUtilization(input);
        expect(result.score).toBe(0);
    });

    it('should return 0 for no response', () => {
        const input: RAGInput = {
            query: 'What is AI?',
            response: '',
            context: ['AI context...']
        };

        const result = contextUtilization(input);
        expect(result.score).toBe(0);
    });
});

describe('RAG Metrics Configuration', () => {
    it('should respect custom threshold', () => {
        const input: RAGInput = {
            query: 'What is AI?',
            response: 'AI is artificial intelligence.',
            context: ['Artificial intelligence is AI.']
        };

        const strictResult = faithfulness(input, { threshold: 0.9 });
        const lenientResult = faithfulness(input, { threshold: 0.1 });

        expect(lenientResult.passed).toBe(true);
        // Strict might fail depending on exact score
    });

    it('should respect similarity threshold', () => {
        const input: RAGInput = {
            query: 'What is ML?',
            response: 'ML is machine learning.',
            context: ['Machine learning processes data.']
        };

        const strictResult = contextPrecision(input, { similarityThreshold: 0.9 });
        const lenientResult = contextPrecision(input, { similarityThreshold: 0.1 });

        expect(lenientResult.score).toBeGreaterThanOrEqual(strictResult.score);
    });
});
