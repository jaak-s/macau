import unittest
import macau

class TestBPMF(unittest.TestCase):
    def test_one(self):
        self.assertTrue( macau.test() > 0 )

if __name__ == '__main__':
    unittest.main()
