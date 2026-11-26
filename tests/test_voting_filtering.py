from gecalgo.voting import consensus_correction, hypothesis_weights

def test_single_beam():
    res=consensus_correction('I go home'.split(), ['I went home'.split()], [0.0],tau=0.5)
    assert res.tokens==['I','went','home']

def test_filter_candidates_subset_of_top1():
    src='I go home'.split()
    hs=['I went home'.split(),'I go home'.split()]
    result=consensus_correction(src,hs,[0.,0.],tau=.6,weight_mode='uniform',restrict_to_top1=True)
    assert result.tokens==src

def test_weights():
    assert abs(sum(hypothesis_weights([1.,2.,3.],temperature=.1))-1)<1e-9

def test_rejects_mismatched_scores():
    import pytest
    with pytest.raises(ValueError):
        consensus_correction('I go home'.split(), ['I went home'.split(), 'I go home'.split()], [0.0])
    with pytest.raises(ValueError):
        consensus_correction('I go home'.split(), [], [])

def test_output_names_do_not_collide():
    from run_voting_filtering import fmt_num, fmt_tau
    assert [fmt_tau(t) for t in (0.5, 0.6, 1.0)] == ['0.5', '0.6', '1.0']   # historical names kept
    assert fmt_tau(0.55) != fmt_tau(0.6)
    assert [fmt_num(t) for t in (0.1, 0.3, 1.0)] == ['0.1', '0.3', '1']   # soft0.1 / soft0.3 / soft1
    assert fmt_num(0.05) == '0.05' and fmt_num(10.0) == '10'
