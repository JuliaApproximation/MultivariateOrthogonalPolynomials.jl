using MultivariateOrthogonalPolynomials, ArrayLayouts, BandedMatrices, Test
import MultivariateOrthogonalPolynomials: ModalInterlace, ModalInterlaceLayout

@testset "ModalInterlace" begin
    ops = [brand(2,3,1,2), brand(1,2,1,1), brand(1,2,1,2)]
    A = ModalInterlace(ops, (3,5), (2,4))
    @test MemoryLayout(A) isa ModalInterlaceLayout
    @test A[[1,4],[1,4,11]] == ops[1]
    @test A[[2],[2,7]] == ops[2]
    @test A[[5],[5,12]] == A[[6],[6,13]] == ops[3]
end