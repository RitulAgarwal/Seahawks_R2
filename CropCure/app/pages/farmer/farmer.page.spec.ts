import { ComponentFixture, TestBed } from '@angular/core/testing';
import { FarmerPage } from './farmer.page';

describe('FarmerPage', () => {
  let component: FarmerPage;
  let fixture: ComponentFixture<FarmerPage>;

  beforeEach(async(() => {
    fixture = TestBed.createComponent(FarmerPage);
    component = fixture.componentInstance;
    fixture.detectChanges();
  }));

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
