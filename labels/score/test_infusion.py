import unittest

from infusion import day_doses


def fragment(start, end, dose=.1, known_day=0):
    return start, end, dose, known_day


class DayDoses(unittest.TestCase):
    def test_sixty_minutes_qualifies(self):
        doses = day_doses([fragment(0, 60)], 0)
        self.assertEqual(doses[0], .1)
        self.assertEqual(doses[1:].sum(), 0)

    def test_fifty_nine_minutes_does_not(self):
        self.assertIsNone(day_doses([fragment(0, 59)], 0))

    def test_touching_fragments_are_one_run(self):
        self.assertIsNotNone(day_doses([fragment(0, 30), fragment(30, 60)], 0))

    def test_overlapping_fragments_are_one_run(self):
        self.assertIsNotNone(day_doses([fragment(0, 40), fragment(30, 60)], 0))

    def test_gap_ends_a_run(self):
        self.assertIsNone(day_doses([fragment(0, 30), fragment(31, 61)], 0))

    def test_separate_short_infusions_do_not_add_up(self):
        self.assertIsNone(day_doses([fragment(0, 30), fragment(300, 330), fragment(600, 630)], 0))

    def test_short_infusion_gives_no_dose_beside_a_long_one(self):
        doses = day_doses([fragment(0, 20, dose=.9), fragment(300, 420, dose=.1)], 0)
        self.assertEqual(doses.max(), .1)
        self.assertEqual(doses[0], 0)

    def test_run_counts_only_once_it_reaches_an_hour_by_the_day_end(self):
        crossing = [fragment(1410, 1470)]
        self.assertIsNone(day_doses(crossing, 0))
        self.assertEqual(day_doses(crossing, 1)[0], .1)

    def test_run_longer_than_an_hour_before_the_day_end_counts_on_both_days(self):
        crossing = [fragment(1320, 1500)]
        self.assertEqual(day_doses(crossing, 0)[23], .1)
        self.assertEqual(day_doses(crossing, 1)[0], .1)

    def test_run_that_ended_on_an_earlier_day_gives_nothing(self):
        self.assertIsNone(day_doses([fragment(0, 120)], 1))

    def test_record_stored_later_is_unknown(self):
        self.assertIsNone(day_doses([fragment(0, 120, known_day=1)], 0))

    def test_record_stored_later_still_extends_a_run_once_known(self):
        run = [fragment(1400, 1440, known_day=1), fragment(1440, 1470, known_day=1)]
        self.assertEqual(day_doses(run, 1)[0], .1)

    def test_unknown_dose_counts_as_use_with_no_dose(self):
        doses = day_doses([fragment(0, 60, dose=0.)], 0)
        self.assertIsNotNone(doses)
        self.assertEqual(doses.sum(), 0)

    def test_hours_keep_the_highest_dose(self):
        doses = day_doses([fragment(0, 30, dose=.1), fragment(30, 90, dose=.3)], 0)
        self.assertEqual(doses[0], .3)
        self.assertEqual(doses[1], .3)


if __name__ == "__main__":
    unittest.main()
